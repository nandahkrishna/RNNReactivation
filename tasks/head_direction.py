# Based on https://github.com/bsorsch/grid_cell_rnn and https://github.com/RatInABox-Lab/RatInABox
import numpy as np
import scipy
import torch

from .task import Task


class HeadDirection(Task):
    """Head direction estimation task (integrate angular velocity cues to estimate head direction).

    Attributes
    ----------
    init_hd : str
        The method of initializing head direction.
    biased : bool
        Whether all random turns are counter-clockwise.
    dt : float
        The time step increment in seconds.
    mu : float
        The turn angle bias in radians.
    sigma : float
        The standard deviation of rotation velocity in radians/second.
    use_hd_cells : bool
        Whether task outputs are the activations of simulated head direction cells.
    hd_cells : HeadDirectionCells or None
        Simulated head direction cells.
    sequence_length : int
        The length of each generated trajectory.
    batch_size : int
        The number of samples in each generated batch.
    device : str
        The device to which tensors will be allocated.

    Methods
    -------
    generate_samples()
        Generate a batch of simulated head direction trajectories.

    get_generator()
        Retrieve a generator that yields batches of head direction trajectories.

    get_test_batch()
        Retrieve a batch of test data for the task.

    decode_outputs(representations)
        Decode outputs from cell-encoded representations.

    compute_metrics(outputs, targets, aux=None)
        Compute evaluation metrics for the task.

    set_device(device='cuda')
        Set the device to allocate tensors to.
    """

    def __init__(
        self,
        init_hd="uniform",
        biased=False,
        dt=0.02,
        mu=0,
        sigma=11.52,
        use_hd_cells=True,
        hd_cells_num=512,
        hd_cells_angular_spread=np.pi / 6,
        sequence_length=100,
        batch_size=200,
        device="cuda",
    ):
        """Constructor for the HeadDirection class.

        Parameters
        ----------
        init_hd : str, optional (default: 'uniform')
            The method of initializing head direction.
            Choices are 'uniform' for a uniform distribution or 'zero' for zero radians.
        biased : bool, optional (default: False)
            If True, all random turns are counter-clockwise.
        dt : float, optional (default: 0.02)
            The time step increment in seconds.
        mu : float, optional (default: 0)
            The turn angle bias in radians.
        sigma : float, optional (default: 11.52)
            The standard deviation of rotation velocity in radians/second.
        use_hd_cells : bool, optional (default: True)
            If True, task outputs are the activations of simulated head direction cells.
        hd_cells_num : int, optional (default: 512)
            The number of head direction cells to use.
        hd_cells_angular_spread : float, optional (default: π/6)
            The angular spread of each head direction cell in radians.
        sequence_length : int, optional (default: 100)
            The length of each generated trajectory.
        batch_size : int, optional (default: 200)
            The number of samples in each generated batch.
        device : str, optional (default: 'cuda')
            The device to which tensors will be allocated (e.g., 'cpu', 'cuda').

        Returns
        -------
        None
        """
        self.init_hd = init_hd
        self.biased = biased
        self.dt = dt
        self.mu = mu
        self.sigma = sigma
        self.use_hd_cells = use_hd_cells
        self.hd_cells = None
        if self.use_hd_cells:
            self.hd_cells = HeadDirectionCells(
                num_cells=hd_cells_num,
                angular_spread=hd_cells_angular_spread,
                device=device,
            )
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.device = device

    def generate_samples(self):
        """Generate a batch of simulated head direction trajectories.

        Returns
        -------
        dict
            A dictionary containing:
            - 'init_hd': The initial head direction for each trajectory.
            - 'ang_v': The angular velocities across timesteps for each trajectory.
            - 'target_hd': The target head directions across timesteps for each trajectory.
        """
        samples = self.sequence_length

        # Initialize variables
        head_dir = np.zeros([self.batch_size, samples + 1])

        if self.init_hd == "uniform":
            head_dir[:, 0] = np.random.uniform(0, 2 * np.pi, self.batch_size)
        else:
            head_dir[:, 0] = np.zeros(self.batch_size)

        ang_velocity = np.zeros([self.batch_size, samples])
        updates = np.zeros([self.batch_size, samples])

        # Generate sequence of random turns
        random_turn = np.random.normal(self.mu, self.sigma, [self.batch_size, samples])
        if self.biased:
            random_turn = np.abs(random_turn)

        for t in range(samples):
            ang_velocity[:, t] = self.dt * random_turn[:, t]
            update = ang_velocity[:, t]
            updates[:, t] = update
            head_dir[:, t + 1] = head_dir[:, t] + update

        head_dir = (
            np.mod(head_dir + np.pi, 2 * np.pi) - np.pi
        )  # Periodic variable, modify range to [-π, π]

        traj = {}

        # Input variables
        traj["init_hd"] = torch.from_numpy(head_dir[:, 0, None]).float().to(self.device)
        traj["ang_v"] = torch.from_numpy(updates[:, :, None]).float().to(self.device)

        # Target variables
        traj["target_hd"] = torch.from_numpy(head_dir[:, 1:, None]).float().to(self.device)

        return traj

    def get_generator(self):
        """Retrieve a generator that yields batches of head direction trajectories.

        Yields
        ------
        dict
            A dictionary containing:
            - 'data': The angular velocity data.
            - 'init_state': The initial state.
            - 'targets': The target outputs.
            - 'init_hd': The initial head direction in radians.
            - 'target_hd': The target head direction in radians.
        """
        while True:
            traj = self.generate_samples()

            ang_v = traj["ang_v"]
            hd = traj["target_hd"]
            init_hd = traj["init_hd"].unsqueeze(-1)

            batch = {
                "data": ang_v,
                "init_state": init_hd,
                "targets": hd,
                "init_hd": init_hd,
                "target_hd": hd,
            }

            if self.use_hd_cells:
                hd_outputs = self.hd_cells.get_activation(hd)
                init_act = torch.squeeze(self.hd_cells.get_activation(init_hd))
                batch = {
                    "data": ang_v,
                    "init_state": init_act,
                    "targets": hd_outputs,
                    "init_hd": init_hd,
                    "target_hd": hd,
                }

            yield batch

    def get_test_batch(self):
        """Retrieve a batch of test data for the task.

        Returns
        -------
        dict
            A dictionary containing the same keys as in `get_generator()`.
        """
        traj = self.generate_samples()

        ang_v = traj["ang_v"]
        hd = traj["target_hd"]
        init_hd = traj["init_hd"].unsqueeze(-1)

        batch = {
            "data": ang_v,
            "init_state": init_hd,
            "targets": hd,
            "init_hd": init_hd,
            "target_hd": hd,
        }

        if self.use_hd_cells:
            hd_outputs = self.hd_cells.get_activation(hd)
            init_act = torch.squeeze(self.hd_cells.get_activation(init_hd))
            batch = {
                "data": ang_v,
                "init_state": init_act,
                "targets": hd_outputs,
                "init_hd": init_hd,
                "target_hd": hd,
            }

        return batch

    def decode_outputs(self, representations):
        """Decode outputs from cell-encoded representations.

        Parameters
        ----------
        representations : torch.Tensor
            Representations (e.g. head direction cell encodings) to decode outputs (e.g. bearing)
            from.

        Returns
        -------
        torch.Tensor
            Decoded outputs.
        """
        if self.use_hd_cells:
            return self.hd_cells.decode_hd(representations)

        return representations

    def compute_metrics(self, outputs, targets, aux=None):
        """Compute evaluation metrics for the task.

        Parameters
        ----------
        outputs : torch.Tensor
            Predicted outputs from a model.
        targets : torch.Tensor
            The true target outputs.
        aux : dict, optional (default: None)
            Auxiliary data for metric computation.

        Returns
        -------
        tuple
            A tuple containing:
            - loss : float
                The computed loss value.
            - metric : dict
                A dictionary of computed metrics.
                Includes the loss and optionally the head direction MSE.
        """
        criterion = torch.nn.MSELoss()
        loss = criterion(outputs, targets)
        metric = {"loss": loss.item()}

        if aux is not None and self.use_hd_cells:
            with torch.no_grad():
                decoded_hd = self.hd_cells.decode_hd(outputs)
                hd_mse = criterion(decoded_hd, aux["target_hd"])
                metric["hd_mse"] = hd_mse.item()

        return loss, metric

    def set_device(self, device="cuda"):
        """Set the device to allocate tensors to.

        Parameters
        ----------
        device : str, optional (default: 'cuda')
            The device to which tensors will be allocated (e.g., 'cpu', 'cuda').

        Returns
        -------
        None
        """
        self.device = device
        if self.use_hd_cells:
            self.hd_cells.set_device(device)


class HeadDirectionCells:
    """Simulated head direction cells.

    This class simulates head direction cell activations for head direction trajectories and can
    decode the head direction in radians from those activations.

    Attributes
    ----------
    num_cells : int
        The number of simulated head direction cells.
    angular_spread : float
        The angular spread of each head direction cell in radians.
    device : str
        The device to which tensors will be allocated.
    us : torch.Tensor
        A tensor of preferred angles in radians for each simulated head direction cell.
    vs : torch.Tensor
        A tensor of angular spreads in radians for each simulated head direction cell.

    Methods
    -------
    get_activation(hd)
        Calculate head direction cell activations for the given head direction trajectories.

    decode_hd(activation, k=3)
        Decode the head direction in radians from head direction cell activations.

    set_device(device)
        Set the device to allocate tensors to.
    """

    def __init__(
        self,
        num_cells=512,
        angular_spread=np.pi / 6,
        device="cuda",
    ):
        """Constructor for the HeadDirectionCells class.

        Parameters
        ----------
        num_cells : int, optional (default: 512)
            The number of head direction cells to simulate.
        angular_spread : float, optional (default: π/6)
            The angular spread of each head direction cell in radians.
        device : str, optional (default: 'cuda')
            The device to which tensors will be allocated (e.g., 'cpu', 'cuda').

        Returns
        -------
        None
        """
        self.num_cells = num_cells
        self.angular_spread = angular_spread
        self.device = device

        self.us = torch.linspace(-np.pi, np.pi, self.num_cells).float().to(self.device)
        self.vs = torch.tensor([angular_spread for _ in range(self.num_cells)]).to(self.device)

    def get_activation(self, hd):
        """Calculate head direction cell activations for the given head direction trajectories.

        Parameters
        ----------
        hd : torch.Tensor
            A tensor of head direction trajectories for which to compute activations.

        Returns
        -------
        torch.Tensor
            A tensor of simulated head direction cell activations.
        """
        outputs = von_mises(hd, self.us, self.vs, norm=1)
        return outputs

    def decode_hd(self, activation, k=3):
        """Decode the head direction in radians from head direction cell activations.

        Parameters
        ----------
        activation : torch.Tensor
            A tensor of head direction cell activations.
        k : int, optional (default: 3)
            The number of top active cells to consider when decoding head direction.

        Returns
        -------
        torch.Tensor
            A tensor containing the decoded head directions.
        """
        idxs = torch.topk(activation.cpu(), k=k)[1].detach().numpy()
        pred_hd = np.take(self.us.cpu().detach().numpy(), idxs, axis=0)
        pred_cos = np.cos(pred_hd).mean(axis=-1)
        pred_sin = np.sin(pred_hd).mean(axis=-1)
        pred_hd = np.arctan2(pred_sin, pred_cos)
        return torch.from_numpy(pred_hd).unsqueeze(-1).float().to(self.device)

    def set_device(self, device="cuda"):
        """Set the device to allocate tensors to.

        Parameters
        ----------
        device : str, optional (default: 'cuda')
            The device to which tensors will be allocated (e.g., 'cpu', 'cuda').

        Returns
        -------
        None
        """
        self.device = device
        self.us.to(device)
        self.vs.to(device)


def von_mises(theta, mu, sigma, norm=None):
    """Probability density function of the von Mises distribution.

    Parameters
    ----------
    theta : torch.Tensor
        A tensor of angle(s) in radians to evaluate the density at.
    mu : torch.Tensor
        The mean direction(s) in radians.
    sigma : float or torch.Tensor
        The standard deviation(s) in radians, which is converted to the spread parameter,
        kappa = 1 / sigma^2. This approximation is only true for sigma << 2 * np.pi.
    norm : float or None, optional (default: None)
        If provided, the maximum value will be the norm.

    Returns
    -------
    torch.Tensor
        A tensor of probability densities, evaluated at the angles provided in `theta`.

    Notes
    -----
    1. Parameters `theta`, `mu` and `sigma` can be any shape as long as they are all the same
       (or strictly, all broadcastable).
    2. All angles must be provided in radians.
    """
    kappa = 1 / (sigma**2)
    v = torch.exp(kappa * torch.cos(theta - mu))
    norm = norm or (torch.exp(kappa) / (2 * np.pi * scipy.special.i0(kappa)))
    norm = norm / torch.exp(kappa)
    v = v * norm
    return v

# Based on https://github.com/bsorsch/grid_cell_rnn
import numpy as np
import torch

from .task import Task


class SpatialNavigation(Task):
    """Spatial position estimation task (integrate velocity cues to estimate position in space).

    Attributes
    ----------
    box_width : float
        The width of the rectangular box environment in meters.
    box_height : float
        The height of the rectangular box environment in meters.
    border_region : float
        The region near the walls where special behavior (slowing down and turning) is triggered.
    border_slow_factor : float
        The factor by which velocity is reduced near the walls.
    init_pos : str
        The method for initializing position.
    biased : bool
        Whether trajectories drift towards/around a specified anchor point.
    drift_const : float
        The strength of the drift when biased.
    anchor_point : np.array
        The anchor point towards/around which trajectories drift when biased.
    dt : float
        The time step increment in seconds.
    mu : float
        The turn angle bias in radians.
    sigma : float
        The standard deviation of rotation velocity in radians/second.
    b : float
        The scale parameter for the Rayleigh distribution of velocity in meters/second.
    use_place_cells : bool
        Whether task outputs are the activations of simulated place cells.
    place_cells : PlaceCells or None
        Simulated place cells.
    sequence_length : int
        The length of each generated trajectory.
    batch_size : int
        The number of samples in each generated batch.
    device : str
        The device to which tensors will be allocated.

    Methods
    -------
    _avoid_wall(position, hd)
        Compute the distance and angle to the nearest wall and determine any necessary turning.

    generate_samples()
        Generate a batch of simulated navigation trajectories.

    get_generator()
        Retrieve a generator that yields batches of navigation trajectories.

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
        box_width=2.2,
        box_height=2.2,
        border_region=0.03,
        border_slow_factor=0.25,
        init_pos="uniform",
        biased=False,
        drift_const=0.05,
        anchor_point=np.array([0, 0]),
        dt=0.02,
        mu=0,
        sigma=11.52,
        b=0.26 * np.pi,
        use_place_cells=True,
        place_cells_num=512,
        place_cells_sigma=0.2,
        place_cells_dog=False,
        place_cells_surround_scale=2,
        sequence_length=100,
        batch_size=200,
        device="cuda",
    ):
        """Constructor for the SpatialNavigation class.

        Parameters
        ----------
        box_width : float, optional (default: 2.2)
            The width of the rectangular box environment in meters.
        box_height : float, optional (default: 2.2)
            The height of the rectangular box environment in meters.
        border_region : float, optional (default: 0.03)
            The region near the walls where trajectories slow down and turn away.
        border_slow_factor : float, optional (default: 0.25)
            The factor by which velocity is reduced near the walls.
        init_pos : str, optional (default: 'uniform')
            The method for initializing position.
            Choices are 'uniform' for a uniform distribution or 'zero' for (0, 0).
        biased : bool, optional (default: False)
            If True, trajectories drift towards/around a specified anchor point.
        drift_const : float, optional (default: 0.05)
            The strength of the drift when biased.
        anchor_point : np.array, optional (default: np.array([0, 0]))
            The anchor point towards/around which trajectories drift when biased.
        dt : float, optional (default: 0.02)
            The time step increment in seconds.
        mu : float, optional (default: 0)
            The turn angle bias in radians.
        sigma : float, optional (default: 11.52)
            The standard deviation of rotation velocity in radians/second.
        b : float, optional (default: 0.26 * np.pi)
            The scale parameter for the Rayleigh distribution of velocity in meters/second.
        use_place_cells : bool, optional (default: True)
            If True, outputs are activations of simulated place cells.
        place_cells_num : int, optional (default: 512)
            The number of place cells to simulate.
        place_cells_sigma : float, optional (default: 0.2)
            The standard deviation of the Gaussian place cell tuning curves.
        place_cells_dog : bool, optional (default: False)
            If True, uses a difference of Gaussians tuning curve for place cells.
        place_cells_surround_scale : float, optional (default: 2)
            The ratio of sigma_2^2 to sigma_1^2 for DoG place cells.
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
        self.box_width = box_width
        self.box_height = box_height
        self.border_region = border_region
        self.border_slow_factor = border_slow_factor
        self.init_pos = init_pos
        self.biased = biased
        self.drift_const = drift_const
        self.anchor_point = anchor_point
        self.dt = dt
        self.mu = mu
        self.sigma = sigma
        self.b = b
        self.use_place_cells = use_place_cells
        self.place_cells = None
        if self.use_place_cells:
            self.place_cells = PlaceCells(
                num_cells=place_cells_num,
                sigma=place_cells_sigma,
                diff_of_gaussians=place_cells_dog,
                surround_scale=place_cells_surround_scale,
                box_width=self.box_width,
                box_height=self.box_height,
                device=device,
            )
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.device = device

    def _avoid_wall(self, position, hd):
        """Compute the distance and angle to the nearest wall and determine any necessary turning.

        Parameters
        ----------
        position : torch.Tensor
            A batch of current positions.
        hd : torch.Tensor
            A batch of current head directions.

        Returns
        -------
        tuple
            A tuple containing:
            - is_near_wall : np.ndarray
                A Boolean array indicating if each position is close a wall.
            - turn_angle : np.ndarray
                The computed angle to turn if near a wall.
        """
        x = position[:, 0]
        y = position[:, 1]
        dists = [
            self.box_width / 2 - x,
            self.box_height / 2 - y,
            self.box_width / 2 + x,
            self.box_height / 2 + y,
        ]
        d_wall = np.min(dists, axis=0)
        angles = np.arange(4) * np.pi / 2
        theta = angles[np.argmin(dists, axis=0)]
        hd = np.mod(hd, 2 * np.pi)
        a_wall = hd - theta
        a_wall = np.mod(a_wall + np.pi, 2 * np.pi) - np.pi

        is_near_wall = (d_wall < self.border_region) * (np.abs(a_wall) < np.pi / 2)
        turn_angle = np.zeros_like(hd)
        turn_angle[is_near_wall] = np.sign(a_wall[is_near_wall]) * (
            np.pi / 2 - np.abs(a_wall[is_near_wall])
        )

        return is_near_wall, turn_angle

    def generate_samples(self):
        """Generate a batch of simulated navigation trajectories.

        Returns
        -------
        dict
            A dictionary containing:
            - 'init_hd': The initial head direction for each trajectory.
            - 'init_x': The initial x position for each trajectory.
            - 'init_y': The initial y position for each trajectory.
            - 'ego_v': The egocentric velocity for each timestep.
            - 'v': The velocity updates for each timestep.
            - 'phi_x': The cosine of the angular velocity.
            - 'phi_y': The sine of the angular velocity.
            - 'target_hd': The target head direction for each timestep.
            - 'target_x': The target x position for each timestep.
            - 'target_y': The target y position for each timestep.
        """
        samples = self.sequence_length

        # Initialize variables
        position = np.zeros([self.batch_size, samples + 2, 2])
        head_dir = np.zeros([self.batch_size, samples + 2])

        if self.init_pos == "uniform":
            position[:, 0, 0] = np.random.uniform(
                -self.box_width / 2, self.box_width / 2, self.batch_size
            )
            position[:, 0, 1] = np.random.uniform(
                -self.box_height / 2, self.box_height / 2, self.batch_size
            )
        else:
            position[:, 0, 0] = np.zeros(self.batch_size)
            position[:, 0, 1] = np.zeros(self.batch_size)

        head_dir[:, 0] = np.random.uniform(0, 2 * np.pi, self.batch_size)
        velocity = np.zeros([self.batch_size, samples + 2])
        updates = np.zeros([self.batch_size, samples + 2, 2])

        # Generate sequence of random boosts and turns
        random_turn = np.random.normal(self.mu, self.sigma, [self.batch_size, samples + 1])
        random_vel = np.random.rayleigh(self.b, [self.batch_size, samples + 1])
        v = np.abs(np.random.normal(0, self.b * np.pi / 2, self.batch_size))

        for t in range(samples + 1):
            # Update velocity
            v = random_vel[:, t]
            turn_angle = np.zeros(self.batch_size)

            # If in border region, turn and slow down
            is_near_wall, turn_angle = self._avoid_wall(position[:, t], head_dir[:, t])
            v[is_near_wall] *= self.border_slow_factor

            # Update turn angle
            turn_angle += self.dt * random_turn[:, t]

            # Take a step
            velocity[:, t] = v * self.dt
            update = velocity[:, t, None] * np.stack(
                [np.cos(head_dir[:, t]), np.sin(head_dir[:, t])], axis=-1
            )

            if self.biased:
                update += self.drift_const * (self.anchor_point - position[:, t])

            updates[:, t] = update
            position[:, t + 1] = position[:, t] + update

            # Rotate head direction
            head_dir[:, t + 1] = head_dir[:, t] + turn_angle

        head_dir = (
            np.mod(head_dir + np.pi, 2 * np.pi) - np.pi
        )  # Periodic variable, modify range to [-π, π]

        traj = {}

        # Input variables
        traj["init_hd"] = torch.from_numpy(head_dir[:, 0, None]).float().to(self.device)
        traj["init_x"] = torch.from_numpy(position[:, 1, 0, None]).float().to(self.device)
        traj["init_y"] = torch.from_numpy(position[:, 1, 1, None]).float().to(self.device)

        traj["ego_v"] = torch.from_numpy(velocity[:, 1:-1]).float().to(self.device)
        traj["v"] = torch.from_numpy(updates[:, 1:-1]).float().to(self.device)
        ang_v = np.diff(head_dir, axis=-1)
        traj["phi_x"] = torch.from_numpy(np.cos(ang_v)[:, :-1]).float().to(self.device)
        traj["phi_y"] = torch.from_numpy(np.sin(ang_v)[:, :-1]).float().to(self.device)

        # Target variables
        traj["target_hd"] = torch.from_numpy(head_dir[:, 1:-1]).float().to(self.device)
        traj["target_x"] = torch.from_numpy(position[:, 2:, 0]).float().to(self.device)
        traj["target_y"] = torch.from_numpy(position[:, 2:, 1]).float().to(self.device)

        return traj

    def get_generator(self):
        """Retrieve a generator that yields batches of navigation trajectories.

        Yields
        ------
        dict
            A dictionary containing:
            - 'data': The velocity data.
            - 'init_state': The initial state.
            - 'targets': The target outputs.
            - 'init_pos': The initial (x, y) positions.
            - 'target_pos': The target (x, y) positions.
        """
        while True:
            traj = self.generate_samples()

            v = traj["v"]
            pos = torch.stack([traj["target_x"], traj["target_y"]], axis=-1)
            init_pos = torch.stack([traj["init_x"], traj["init_y"]], axis=-1)
            batch = {
                "data": v,
                "init_state": init_pos,
                "targets": pos,
                "init_pos": init_pos,
                "target_pos": pos,
            }

            if self.use_place_cells:
                place_outputs = self.place_cells.get_activation(pos)
                init_act = self.place_cells.get_activation(init_pos)
                batch = {
                    "data": v,
                    "init_state": init_act,
                    "targets": place_outputs,
                    "init_pos": init_pos,
                    "target_pos": pos,
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

        v = traj["v"]
        pos = torch.stack([traj["target_x"], traj["target_y"]], axis=-1)
        init_pos = torch.stack([traj["init_x"], traj["init_y"]], axis=-1)

        batch = {
            "data": v,
            "init_state": init_pos,
            "targets": pos,
            "init_pos": init_pos,
            "target_pos": pos,
        }

        if self.use_place_cells:
            place_outputs = self.place_cells.get_activation(pos)
            init_act = torch.squeeze(self.place_cells.get_activation(init_pos))
            batch = {
                "data": v,
                "init_state": init_act,
                "targets": place_outputs,
                "init_pos": init_pos,
                "target_pos": pos,
            }

        return batch

    def decode_outputs(self, representations):
        """Decode outputs from cell-encoded representations.

        Parameters
        ----------
        representations : torch.Tensor
            Representations (e.g. place cell encodings) to decode outputs (e.g. position) from.

        Returns
        -------
        torch.Tensor
            Decoded outputs.
        """
        if self.use_place_cells:
            return self.place_cells.decode_pos(representations)

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
                Includes the loss and optionally the position MSE.
        """
        criterion = torch.nn.MSELoss()
        loss = criterion(outputs, targets)
        metric = {"loss": loss.item()}

        if aux is not None:
            with torch.no_grad():
                decoded_pos = self.place_cells.decode_pos(outputs)
                pos_mse = criterion(decoded_pos, aux["target_pos"])
                metric["pos_mse"] = pos_mse.item()

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
        if self.use_place_cells:
            self.place_cells.set_device(device)


class PlaceCells:
    """Simulated place cells.

    This class simulates place cell activations for navigation trajectories in a box environment.
    It can also decode positions from the activations of these cells.

    Attributes
    ----------
    num_cells : int
        The number of simulated place cells.
    sigma : float
        The standard deviation for the Gaussian place cell tuning curves.
    diff_of_gaussians : bool
        Whether to use a difference of Gaussians tuning curve for place cells.
    surround_scale : float
        The ratio of sigma_2^2 to sigma_1^2 for DoG place cells.
    box_width : float
        The width of the environment in meters.
    box_height : float
        The height of the environment in meters.
    device : str
        The device to which tensors will be allocated.
    us : torch.Tensor
        A tensor of simulated place cell centers.

    Methods
    -------
    get_activation(pos)
        Get place cell activations for the given navigation trajectories.

    decode_pos(activation, k=3)
        Decode position from place cell activations.

    set_device(device)
        Set the device to allocate tensors to.
    """

    def __init__(
        self,
        num_cells=512,
        sigma=0.2,
        diff_of_gaussians=False,
        surround_scale=2,
        box_width=2.2,
        box_height=2.2,
        device="cuda",
    ):
        """Constructor for the PlaceCells class.

        Parameters
        ----------
        num_cells : int, optional (default: 512)
            The number of place cells to simulate.
        sigma : float, optional (default: 0.2)
            The standard deviation for the Gaussian place cell tuning curves.
        diff_of_gaussians : bool, optional (default: False)
            If True, uses a difference of Gaussians tuning curve for place cells.
        surround_scale : float, optional (default: 2)
            The ratio of sigma_2^2 to sigma_1^2 for DoG place cells.
        box_width : float, optional (default: 2.2)
            The width of the environment in meters.
        box_height : float, optional (default: 2.2)
            The height of the environment in meters.
        device : str, optional (default: 'cuda')
            The device to which tensors will be allocated (e.g., 'cpu' or 'cuda').

        Returns
        -------
        None
        """
        self.num_cells = num_cells
        self.sigma = sigma
        self.diff_of_gaussians = diff_of_gaussians
        self.surround_scale = surround_scale
        self.box_width = box_width
        self.box_height = box_height
        self.device = device

        # Randomly tile place cell centers across environment
        usx = np.random.uniform(-self.box_width / 2, self.box_width / 2, (self.num_cells,))
        usy = np.random.uniform(-self.box_height / 2, self.box_height / 2, (self.num_cells,))
        self.us = torch.from_numpy(np.stack([usx, usy], axis=-1)).float().to(self.device)

    def get_activation(self, pos):
        """Get place cell activations for the given navigation trajectories.

        Parameters
        ----------
        pos : torch.Tensor
            A tensor of navigation trajectories for which to compute place cell activations.

        Returns
        -------
        torch.Tensor
            A tensor of simulated place cell activations.
        """
        d = torch.abs(pos[:, :, None, :] - self.us[None, None, ...])
        norm2 = torch.sum(d**2, axis=-1)

        # Normalize with softmax (nearly equivalent to using prefactor)
        outputs = torch.softmax(-norm2 / (2 * self.sigma**2), dim=-1)

        if self.diff_of_gaussians:
            # Normalize again with softmax
            outputs -= torch.softmax(-norm2 / (2 * self.surround_scale * self.sigma**2), dim=-1)

            # Shift and scale outputs so that they lie in [0, 1]
            outputs += torch.abs(torch.min(outputs, dim=-1, keepdim=True))
            outputs /= torch.sum(outputs, dim=-1, keepdim=True)

        return outputs

    def decode_pos(self, activation, k=3):
        """Decode position from place cell activations.

        Parameters
        ----------
        activation : torch.Tensor
            A tensor of place cell activations.

        k : int, optional (default: 3)
            The number of top active cells to consider when decoding position.

        Returns
        -------
        torch.Tensor
            A tensor containing the decoded positions.
        """
        idxs = torch.topk(activation.cpu(), k=k)[1].detach().numpy()
        pred_pos = np.mean(np.take(self.us.cpu().detach().numpy(), idxs, axis=0), axis=-2)
        return torch.from_numpy(pred_pos).float().to(self.device)

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

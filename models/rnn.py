import torch
import torch.nn as nn


class RNN(nn.Module):
    """Vanilla continuous-time recurrent neural network (RNN) with linear readouts.

    Attributes
    ----------
    task : Task
        The task for which the RNN is used.
    n_in : int
        The number of input features.
    n_rec : int
        The number of recurrent units.
    n_out : int
        The number of output units.
    n_init : int
        The dimensionality of the initial state.
    sigma_in : float
        The standard deviation of input noise.
    sigma_rec : float
        The standard deviation of recurrent noise.
    sigma_out : float
        The standard deviation of output noise.
    dt : float
        The time step for integration.
    tau : float
        The time constant of the RNN units.
    bias : bool
        Whether biases are used in Linear layers.
    activation_fn : str
        The activation function used for the RNN units.
    device : str
        The device on which the RNN is initialized.
    encoder : torch.nn.Linear
        A Linear layer for encoding the initial state.
    w_in : torch.nn.Linear
        A Linear layer for the input weights.
    w_rec : torch.nn.Linear
        A Linear layer for the recurrent weights.
    w_out : torch.nn.Linear
        A Linear layer for the output weights.
    activation : function or torch.nn.Module
        The activation function chosen based on `activation_fn`.

    Methods
    -------
    forward(x, init_state=None)
        Perform the forward pass through the RNN.
    """

    def __init__(
        self,
        task,
        n_in=2,
        n_rec=512,
        n_out=512,
        n_init=512,
        sigma_in=0,
        sigma_rec=0,
        sigma_out=0,
        dt=0.2,
        tau=1,
        bias=False,
        activation_fn="relu",
        device="cuda",
    ):
        """Constructor for the RNN class.

        Parameters
        ----------
        task : Task
            The task for which the RNN is used.
        n_in : int, optional (default: 2)
            The number of input features.
        n_rec : int, optional (default: 512)
            The number of recurrent units.
        n_out : int, optional (default: 512)
            The number of output units.
        n_init : int, optional (default: 512)
            The dimensionality of the initial state.
        sigma_in : float, optional (default: 0)
            The standard deviation of input noise.
        sigma_rec : float, optional (default: 0)
            The standard deviation of recurrent noise.
        sigma_out : float, optional (default: 0)
            The standard deviation of output noise.
        dt : float, optional (default: 0.2)
            The time step for integration.
        tau : float, optional (default: 1)
            The time constant of RNN units.
        bias : bool, optional (default: False)
            If True, use biases in Linear layers.
        activation_fn : str, optional (default: 'relu')
            The activation function to use for the RNN units.
            Choices are 'relu', 'tanh', 'sigmoid', or 'identity'.
        device : str, optional (default: 'cuda')
            The device to run computations on (e.g., 'cpu', 'cuda').

        Returns
        -------
        None
        """
        super(RNN, self).__init__()

        self.task = task
        self.n_in = n_in
        self.n_rec = n_rec
        self.n_out = n_out
        self.n_init = n_init
        self.sigma_in = sigma_in
        self.sigma_rec = sigma_rec
        self.sigma_out = sigma_out
        self.dt = dt
        self.tau = tau
        self.bias = bias
        self.activation_fn = activation_fn
        self.device = device

        self.encoder = nn.Linear(self.n_init, self.n_rec, bias=self.bias)
        self.w_in = nn.Linear(self.n_in, self.n_rec, bias=self.bias)
        self.w_rec = nn.Linear(self.n_rec, self.n_rec, bias=self.bias)
        self.w_out = nn.Linear(self.n_rec, self.n_out, bias=self.bias)

        if self.activation_fn == "relu":
            self.activation = torch.relu
        elif self.activation_fn == "tanh":
            self.activation = torch.tanh
        elif self.activation_fn == "sigmoid":
            self.activation = torch.sigmoid
        else:
            self.activation = nn.Identity()

        self.to(self.device)

    def forward(self, x, init_state=None):
        """Perform the forward pass through the RNN.

        Parameters
        ----------
        x : torch.Tensor
            An input tensor of shape (batch_size, sequence_length, n_in).
        init_state : torch.Tensor or None, optional (default: None)
            An initial state tensor. If provided, should be of shape (batch_size, n_init).

        Returns
        -------
        tuple
            A tuple containing:
            - h_1t : torch.Tensor
                A tensor of hidden states over time,
                shape (batch_size, sequence_length, n_rec).
            - y_1t : torch.Tensor
                A tensor of outputs over time,
                shape (batch_size, sequence_length, n_out).
        """
        batch_size, timesteps = x.shape[0], x.shape[1]

        if init_state is not None:
            self.h = self.encoder(init_state.reshape(batch_size, self.n_init))
        else:
            self.h = torch.zeros(batch_size, self.n_rec, device=self.device)

        self.h_1t = torch.zeros(batch_size, timesteps, self.n_rec, device=self.device)
        self.y_1t = torch.zeros(batch_size, timesteps, self.n_out, device=self.device)

        for t in range(timesteps):
            x_t = x[:, t, :].reshape(batch_size, self.n_in)
            noise_in = torch.rand_like(x_t, device=self.device)
            self.u = self.w_rec(self.h) + self.w_in(x_t + self.sigma_in * noise_in)

            noise_rec = torch.rand_like(self.h, device=self.device)
            self.h = (
                (1 - self.dt / self.tau) * self.h
                + (self.dt / self.tau) * self.activation(self.u)
                + self.sigma_rec * noise_rec
            )

            noise_out = torch.randn(batch_size, self.n_out, device=self.device)
            self.y = self.w_out(self.h) + self.sigma_out * noise_out

            self.h_1t[:, t, :] = self.h
            self.y_1t[:, t, :] = self.y

        return self.h_1t, self.y_1t
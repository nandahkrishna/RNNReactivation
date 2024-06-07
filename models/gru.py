import torch
import torch.nn as nn


class GRU(nn.Module):
    """Continuous-time Gated Recurrent Unit (GRU) with linear readouts.

    Attributes
    ----------
    task : Task
        The task for which the GRU is used.
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
        The time constant of the GRU units.
    bias : bool
        Whether biases are used in Linear layers.
    activation_fn : str
        The activation function used for the GRU units.
    device : str
        The device on which the GRU is initialized.
    encoder : nn.Linear
        A Linear layer for encoding the initial state.
    w_z : torch.nn.Linear
        A Linear layer for the update gate's input weights.
    u_z : torch.nn.Linear
        A Linear layer for the update gate's recurrent weights.
    w_r : torch.nn.Linear
        A Linear layer for the reset gate's input weights.
    u_r : torch.nn.Linear
        A Linear layer for the reset gate's recurrent weights.
    w_h : torch.nn.Linear
        A Linear layer for the hidden state's input weights.
    u_h : torch.nn.Linear
        A Linear layer for the hidden state's recurrent weights.
    decoder : torch.nn.Linear
        A Linear layer for the output weights.
    activation : function or torch.nn.Module
        The activation function chosen based on `activation_fn`.

    Methods
    -------
    forward(x, init_state=None)
        Perform the forward pass through the GRU.
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
        activation_fn="tanh",
        device="cuda",
    ):
        """Constructor for the GRU class.

        Parameters
        ----------
        task : Task
            The task for which the GRU is used.
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
            The time constant of the GRU units.
        bias : bool, optional (default: False)
            If True, use biases in Linear layers.
        activation_fn : str, optional (default: 'tanh')
            The activation function to use for the GRU units.
            Choices are 'relu', 'tanh', 'sigmoid', or 'identity'.
        device : str, optional (default: 'cuda')
            The device to run computations on (e.g., 'cpu', 'cuda').

        Returns
        -------
        None
        """
        super(GRU, self).__init__()

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
        self.w_z = nn.Linear(self.n_in, self.n_rec, bias=self.bias)
        self.u_z = nn.Linear(self.n_rec, self.n_rec, bias=self.bias)
        self.w_r = nn.Linear(self.n_in, self.n_rec, bias=self.bias)
        self.u_r = nn.Linear(self.n_rec, self.n_rec, bias=self.bias)
        self.w_h = nn.Linear(self.n_in, self.n_rec, bias=self.bias)
        self.u_h = nn.Linear(self.n_rec, self.n_rec, bias=self.bias)
        self.decoder = nn.Linear(self.n_rec, self.n_out, bias=self.bias)

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
        """Perform the forward pass through the GRU.

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
            x_t = x_t + self.sigma_in * noise_in

            self.z = torch.sigmoid(self.w_z(x_t) + self.u_z(self.h))

            self.r = torch.sigmoid(self.w_r(x_t) + self.u_r(self.h))

            self.u = self.w_h(x_t) + self.u_h(self.r * self.h)
            noise_rec = torch.rand_like(self.h, device=self.device)
            self.h = (
                self.h
                + (self.dt / self.tau) * ((self.z - 1) * (self.h - self.activation(self.u)))
                + self.sigma_rec * noise_rec
            )

            noise_out = torch.randn(batch_size, self.n_out, device=self.device)
            self.y = self.decoder(self.h) + self.sigma_out * noise_out

            self.h_1t[:, t, :] = self.h
            self.y_1t[:, t, :] = self.y

        return self.h_1t, self.y_1t

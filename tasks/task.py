import abc


class Task(abc.ABC):
    """Abstract base class for defining tasks.

    Methods
    -------
    generate_samples()
        Generate a batch of samples or trajectories for the task.

    get_generator()
        Retrieve a data generator for the task.

    get_test_batch()
        Retrieve a batch of test data for the task.

    decode_outputs(representations)
        Decode outputs from cell-encoded representations.

    compute_metrics(outputs, targets, aux)
        Compute evaluation metrics for the task.

    set_device(device)
        Set the device to allocate tensors to.
    """

    @abc.abstractmethod
    def generate_samples(self):
        """Generate a batch of samples or trajectories for the task."""

    @abc.abstractmethod
    def get_generator(self):
        """Retrieve a data generator for the task."""

    @abc.abstractmethod
    def get_test_batch(self):
        """Retrieve a batch of test data for the task."""

    @abc.abstractmethod
    def decode_outputs(self, representations):
        """Decode outputs from cell-encoded representations."""

    @abc.abstractmethod
    def compute_metrics(self, outputs, targets, aux):
        """Compute evaluation metrics for the task."""

    def set_device(self, device):
        """Set the device to allocate tensors to."""
        self.device = device

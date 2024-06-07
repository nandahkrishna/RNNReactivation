import json
import pathlib
import torch
import torch.optim as optim


class Trainer:
    """Class for training and evaluating a model.

    Attributes
    ----------
    model : torch.nn.Module
        The model to be trained.
    train_data : iterable
        A training data generator for a task.
    test_data : dict
        A batch of test data for a task.
    lr : float
        The learning rate for the Adam optimizer.
    weight_decay : float
        The weight decay for the Adam optimizer.
    compute_all_metrics : bool
        Whether to compute all task metrics.
    test_freq : int
        The frequency (in epochs) of testing the model.
    save_freq : int
        The frequency (in epochs) of saving the model.
    path : str
        The path to save model checkpoints and metrics to.
    device : str
        The device on which to perform computations.
    optimizer : optim.Adam
        An instance of the Adam optimizer for model parameters.

    Methods
    -------
    train(n_epochs=2500, start_epoch=0)
        Train the model for a specified number of epochs.
    """

    def __init__(
        self,
        model,
        train_data,
        test_data,
        lr=0.001,
        weight_decay=0,
        compute_all_metrics=True,
        test_freq=100,
        save_freq=100,
        path="results",
        device="cuda",
    ):
        """Constructor for the Trainer class.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be trained.
        train_data : iterable
            A training data generator for a task.
        test_data : dict
            A batch of test data for a task.
        lr : float, optional (default: 0.001)
            The learning rate for the Adam optimizer.
        weight_decay : float, optional (default: 0)
            The weight decay for the Adam optimizer.
        compute_all_metrics : bool, optional (default: True)
            If True, compute all task metrics during training and testing.
        test_freq : int, optional (default: 100)
            The frequency (in epochs) at which to evaluate the model on the test data.
        save_freq : int, optional (default: 100)
            The frequency (in epochs) at which to save the model.
        path : str, optional (default: "results")
            The directory to save model checkpoints and metrics to.
        device : str, optional (default: 'cuda')
            The device to run computations on (e.g., 'cpu', 'cuda').

        Returns
        -------
        None
        """
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.lr = lr
        self.weight_decay = weight_decay
        self.compute_all_metrics = compute_all_metrics
        self.test_freq = test_freq
        self.save_freq = save_freq
        self.path = path
        self.device = device

        self.model.set_device(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def train(self, n_epochs=2500, start_epoch=0):
        """Train `self.model` for a specified number of epochs.

        Parameters
        ----------
        n_epochs : int, optional (default: 2500)
            The number of epochs to train the model for.
        start_epoch : int, optional (default: 0)
            The starting epoch number (for checkpoint numbering).

        Returns
        -------
        None
        """
        epoch = start_epoch
        self.train_metrics = dict()
        self.test_metrics = dict()
        aux = None

        for batch in self.train_data:
            epoch += 1
            if epoch - start_epoch > n_epochs:
                break

            self.model.train()
            self.optimizer.zero_grad()

            data = batch["data"].to(self.device)
            init_state = batch["init_state"].to(self.device)
            targets = batch["targets"].to(self.device)

            _, outputs = self.model(data, init_state=init_state)

            if self.compute_all_metrics:
                aux = batch

            train_loss, train_metric = self.model.task.compute_metrics(outputs, targets, aux)
            train_loss.backward()
            self.optimizer.step()
            self.train_metrics[epoch] = train_metric.copy()

            print(f"Epoch {epoch} (train):")
            for k, v in train_metric.items():
                print(f"  - {k} = {v}.")

            if epoch % self.save_freq == 0:
                model_path = pathlib.Path(self.path).joinpath(f"model_{epoch}.pt")
                torch.save(self.model, model_path)
                print(f"Model saved at epoch {epoch}.")

            if epoch % self.test_freq == 0:
                with torch.no_grad():
                    self.model.eval()

                    data = self.test_data["data"].to(self.device)
                    init_state = self.test_data["init_state"].to(self.device)
                    targets = self.test_data["targets"].to(self.device)

                    if self.compute_all_metrics:
                        aux = self.test_data

                    _, outputs = self.model(data, init_state=init_state)

                    _, test_metric = self.model.task.compute_metrics(outputs, targets, aux)
                    self.test_metrics[epoch] = test_metric.copy()

                    print(f"Epoch {epoch} (test):")
                    for k, v in test_metric.items():
                        print(f"  - {k} = {v}.")

        train_metrics_path = pathlib.Path(self.path).joinpath("train_metrics.json")
        json.dump(self.train_metrics, open(train_metrics_path, "w"), indent=4)

        test_metrics_path = pathlib.Path(self.path).joinpath("test_metrics.json")
        json.dump(self.test_metrics, open(test_metrics_path, "w"), indent=4)

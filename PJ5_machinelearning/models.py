import nn

class PerceptronModel(object):

    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        return 1 if nn.as_scalar(nn.DotProduct(self.w, x)) >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        while True:
            DONE = True
            for x, y in dataset.iterate_once(1):
                if nn.as_scalar(y) != self.get_prediction(x):
                    DONE = False
                    self.w.update(x, nn.as_scalar(y))
            if DONE:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 50
        self.w0 = nn.Parameter(1, 80)
        self.b0 = nn.Parameter(1, 80)
        self.w1 = nn.Parameter(80, 1)
        self.b1 = nn.Parameter(1, 1)
        self.alpha = 0.0025

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"

        xw1 = nn.Linear(x, self.w0)
        r1 = nn.ReLU(nn.AddBias(xw1, self.b0))
        xw2 = nn.Linear(r1, self.w1)
        return nn.AddBias(xw2, self.b1)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        while True:

            # print(nn.Constant(dataset.x), nn.Constant(dataset.y))

            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                grad = nn.gradients(loss, [self.w0, self.w1, self.b0, self.b1])

                # print(nn.as_scalar(nn.DotProduct(grad[0],grad[0])))
                self.w0.update(grad[0], -self.alpha)
                self.w1.update(grad[1], -self.alpha)
                self.b0.update(grad[2], -self.alpha)
                self.b1.update(grad[3], -self.alpha)

            # print(nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))))
            if nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))) < 1e-4:
                return

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 10
        self.alpha = 0.01
        self.w0 = nn.Parameter(self.num_chars, 300)  # w0
        self.b0 = nn.Parameter(1, 300)  # b0
        self.w1 = nn.Parameter(300, 300)  # w1
        self.b1 = nn.Parameter(1, 300)  # b1
        self.wf = nn.Parameter(300, 5)  # wf
        self.bf = nn.Parameter(1, 5)  # bf

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        h = nn.ReLU(nn.AddBias(nn.Linear(xs[0], self.w0), self.b0))
        z = h
        for i, x in enumerate(xs[1:]):
            z = nn.Add(nn.ReLU(nn.AddBias(nn.Linear(z, self.w1), self.b1)),
                       nn.ReLU(nn.AddBias(nn.Linear(x, self.w0), self.b0)))
            z = nn.Add(nn.ReLU(nn.Linear(z, self.w1)), nn.ReLU(nn.Linear(x, self.w0)))
            # z = nn.Add(nn.Linear(x, self.w0), nn.Linear(z, self.w1))
            # print(i, nn.Linear(z, self.wf))
        return nn.AddBias(nn.Linear(z, self.wf), self.bf)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:

            # print(nn.Constant(dataset.x), nn.Constant(dataset.y))

            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                grad = nn.gradients(loss, [self.w0, self.w1, self.wf, self.b0, self.b1, self.bf])

                # print(grad[0], grad[1], grad[2])
                self.w0.update(grad[0], -self.alpha)
                self.w1.update(grad[1], -self.alpha)
                self.wf.update(grad[2], -self.alpha)
                self.b0.update(grad[3], -self.alpha)
                self.b1.update(grad[4], -self.alpha)
                self.bf.update(grad[5], -self.alpha)

            print("""
            *
            *
            *\n""",
                  dataset.get_validation_accuracy(),
                  """
                  *
                  *
                  *
                  *""")
            if dataset.get_validation_accuracy() >= 0.9:
                return

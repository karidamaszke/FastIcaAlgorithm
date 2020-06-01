import numpy as np
import matplotlib.pyplot as plt


def plot_signals(first_signal, second_signal, title):
    """
    Display part of given signals for visualize algorithm's action.

    :param first_signal: numpy.array
    :param second_signal: numpy.array
    :param title: what is plotted
    """
    samples_count = 2000

    # normalize signals
    first_max = np.max(np.abs(first_signal))
    first_norm_signal = (first_signal / first_max) * 100

    second_max = np.max(np.abs(second_signal))
    second_norm_signal = (second_signal / second_max) * 100

    # plot signals
    plt.figure()

    plt.subplot(2, 1, 1)
    plt.plot(np.linspace(1, samples_count + 1, samples_count), first_norm_signal[samples_count:2 * samples_count])
    plt.title(title)

    plt.subplot(2, 1, 2)
    plt.plot(np.linspace(1, samples_count + 1, samples_count), second_norm_signal[samples_count:2 * samples_count])


def mix_signals(first_signal, second_signal):
    """
    Artificial mixing input signals to create linear mixtures.
    Center data to make its means zero.
    :param first_signal: numpy.array
    :param second_signal: numpy.array

    :return: array 2xN, when N is number of samples
    """
    mix_coefficients = np.random.rand(2, 2)

    first_mix_signal = mix_coefficients[0][0] * first_signal + mix_coefficients[0][1] * second_signal
    second_mix_signal = mix_coefficients[1][0] * first_signal + mix_coefficients[1][1] * second_signal

    """ 
    center data:

    x(t) - mean(x(t))      -> mean(x(t)) is arithmetic mean of signal
    ----------------
        std(x(t))          -> std(x(t)) is standard deviation of signal

    """
    first_mix_signal = np.array((first_mix_signal - np.mean(first_mix_signal)) / np.std(first_mix_signal))
    second_mix_signal = np.array((second_mix_signal - np.mean(second_mix_signal)) / np.std(second_mix_signal))

    return np.c_[[first_mix_signal, second_mix_signal]]


def initialize_w():
    """
    Create vector for separation.
    Initialize with random values.

    :return: vector 2x1, normalized
    """
    w = np.random.rand(2, 1)
    w /= np.sqrt((w ** 2).sum())
    return w


class FastIca:
    """
    Class for demonstration of fast ICA algorithm.
    It creates two mixtures of signals and separates them.
    """

    def __init__(self, first_signal, second_signal, plot=False):
        self.plot = plot
        self.mixed_signal = mix_signals(first_signal, second_signal)
        self.w = initialize_w()

    def run_algorithm(self):
        """
        STEPS:
        1. whiten signals
        2. one source extraction
        3. create vector orthogonal to w
        4. check if w.T * w is equal to 1
        :return: separate signals
        """
        if self.plot:
            plot_signals(self.mixed_signal[0], self.mixed_signal[1], "Mixed signals")

        whitened_signal = self.whiten_data(self.mixed_signal)

        iterations = 0
        while True:
            iterations += 1
            last_w = self.w
            self.w = self.one_source_extraction(whitened_signal)
            v = self.get_orthogonal()
            new_signal = self.source_separation(v, whitened_signal)

            if self.is_end(last_w):
                print("Finished after " + str(iterations) + " iterations.")
                break

            if iterations > 50:
                raise Exception("Algorithm doesn't finished separation after 50 iterations!")

        return 0.4 * new_signal[0], 0.8 * new_signal[1]

    def whiten_data(self, signal_x):
        """
        Whiten the data to obtain uncorrelated mixture signals

        STEPS:
        1. calculate covariance matrix -> cov[X] = E[X X.T]
        2. calculate eigenvalues λ and eigenvectors V
        3. calculate Λ^−0.5 matrix, where Λ is matrix with eigenvalues on diagonal
        4. get eigenvectors W from normalizing V
        5. calculate whiten signal from Y = W * Λ^-0.5 * W.T * X

        :return: whitened signal
        """
        covariance_matrix = np.cov(signal_x)  # 1
        eig_values, eig_vectors = np.linalg.eigh(covariance_matrix)  # 2
        lambda_matrix = np.diag(1. / np.sqrt(eig_values))  # 3
        w_matrix = self.normalize_eig_vectors(eig_vectors)  # 4
        signal_y = np.dot(np.dot(np.dot(w_matrix, lambda_matrix), w_matrix.T), signal_x)  # 5

        return signal_y

    @staticmethod
    def normalize_eig_vectors(eig_vectors):
        """
        Normalize given vector
        """
        for i, _ in enumerate(eig_vectors):
            eig_vectors[i] /= np.sqrt((eig_vectors[i] ** 2).sum())
        return eig_vectors

    def one_source_extraction(self, whiten_signal):
        """
        One iteration of algorithm.

        STEPS:
        1. calculate W.T * mix_signal
        2. get 1. step to power 3
        3. multiply mix_signal with 2. step.
        4. get average vector of 3. step.
        5. calculate new W vector as 4. step - 3 * W
        6. normalize new W vector

        :return: new W vector
        """
        signal = whiten_signal * np.power(np.dot(self.w.T, whiten_signal), 3)  # 1, 2, 3
        average = np.array(np.mean(signal, axis=1)).reshape(-1, 1)  # 4
        new_w = average - 3 * self.w  # 5
        new_w /= np.sqrt((new_w ** 2).sum())  # 6

        return new_w

    def get_orthogonal(self):
        """
        Calculate orthogonal vector for current W vector
        :return: vector V
        """
        v = np.array([self.w[1][0], -self.w[0][0]]).reshape(-1, 1)
        return v

    def source_separation(self, v, whiten_signal):
        """
        Separate signals with new values of W and V vectors
        :param v: vector orthogonal for current W vector
        :param whiten_signal: mixed and whiten signal
        :return: array 2xN, where N is number of samples
        """
        matrix = np.concatenate((self.w.T, v.T))
        new_signal = np.dot(matrix, whiten_signal)

        return new_signal

    def is_end(self, last_w):
        """
        Check ending condition.
        :param last_w: last value of W vector
        :return: bool: is algorithm finished or not
        """
        result = np.dot(self.w.T, last_w)
        if abs(1.0 - result) < 0.000001:
            return True
        if result < -0.5:
            self.w = initialize_w()
        return False

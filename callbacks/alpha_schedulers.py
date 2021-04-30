from drig.utils import log


class AlphaSchedulers:
    def __init__(self, base_alpha, total_epochs):
        try:
            self.base_alpha = base_alpha
            self.total_epochs = total_epochs
        except Exception as e:
            raise e

    def polynomial_decay(self, epoch):
        try:

            exp = 1.0

            new_alpha = self.base_alpha * (
                1 - (epoch / float(self.total_epochs)))**exp
            log.info(f"---SETTING NEW ALPHA : {new_alpha}")
            return new_alpha

        except Exception as e:
            raise e

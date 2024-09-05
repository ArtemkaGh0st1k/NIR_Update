class UpperException(Exception):
    def __init__(self, *args: object) -> None:
        self.args = args
        super().__init__(*args)


class LowerException(Exception):
    def __init__(self, *args: object) -> None:
        self.args = args
        super().__init__(*args)
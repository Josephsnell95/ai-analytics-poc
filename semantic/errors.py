class ModelError(Exception):
    """The YAML model itself is invalid. Raised at load time, aimed at the model author."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("Invalid semantic model:\n- " + "\n- ".join(errors))


class QueryValidationError(Exception):
    """A semantic query does not fit the model.

    Messages are written for the agent to act on: say what was wrong and what the
    valid options are. All problems are collected and reported together, so the
    agent can fix everything in one retry instead of one error per round trip.
    """

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("Invalid semantic query:\n- " + "\n- ".join(errors))

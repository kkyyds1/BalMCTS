from dataclasses import dataclass

@dataclass
class Variable:
    index: int
    domain: list
    is_assigned: int
    constraints: list

@dataclass
class Constraint:
    index: int
    variables: tuple
    relations: list
    product: int
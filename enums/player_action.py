from enum import Enum, IntEnum


class PlayerAction(IntEnum):
    NOP = 0,
    NORTH = 1,
    SOUTH = 2,
    EAST = 3,
    WEST = 4,
    INTERACT = 5,
    TOGGLE = 6,
    CANCEL = 7,
    PICKUP = 8,
    RESET = 9,

PlayerActionTable = {
    "NOP" : PlayerAction.NOP,
    "NORTH" : PlayerAction.NORTH,
    "SOUTH" : PlayerAction.SOUTH,
    "EAST" : PlayerAction.EAST,
    "WEST" : PlayerAction.WEST,
    "INTERACT" : PlayerAction.INTERACT,
    "TOGGLE" : PlayerAction.TOGGLE,
    "CANCEL" : PlayerAction.CANCEL,
    "PICKUP" : PlayerAction.PICKUP,
    "RESET" : PlayerAction.RESET,
}

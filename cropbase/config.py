class CROP_TYPE:
    RICE = 1
    MAIZE = 2
    SOYBEAN = 3
    WHEAT = 4

    dict = {
        RICE: "rice",
        MAIZE: "maize",
        SOYBEAN: "soybean",
        WHEAT: "wheat",
    }

    color_dict = {
        RICE: (157, 210, 16),
        MAIZE: (255, 158, 8),
        SOYBEAN: (244, 94, 25),
        WHEAT: (255, 231, 199)
    }


class DISASTER_TYPE:
    FLOOD = 3
    LODGE = 4
    WATERLOGGING = 1
    DROUGHT = 2
    NODISASTER = 0

    value2text_dict = {
        FLOOD: "flood",
        LODGE: "lodge",
        WATERLOGGING: "waterlogging",
        DROUGHT: "drought",
        NODISASTER: "nodisaster"
    }

    text2value_dict = {
        "flood": FLOOD,
        "lodge": LODGE,
        "waterlogging": WATERLOGGING,
        "drought": DROUGHT,
        "nodisaster": NODISASTER
    }

    color_dict = {
        FLOOD: (127, 100, 92),
        LODGE: (219, 238, 229),
        WATERLOGGING: (219, 193, 154),
        DROUGHT: (0, 0, 0),

    }

from bidict import bidict

ROOM_CLASS = bidict(
    {
        "living_room": 1,
        "kitchen": 2,
        "bedroom": 3,
        "bathroom": 4,
        "balcony": 5,
        "entrance": 6,
        "dining room": 7,
        "study room": 8,
        "storage": 10,
        "front door": 15,
        "unknown": 16,
        "interior_door": 17,
    }
)

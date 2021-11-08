import numpy as np
from PIL import Image, ImageDraw


def bb_to_vec(bbs_batch):
    """This functions seems to convert bbox to lines"""
    cs_type_batch = []
    for bbs in bbs_batch:
        corners_set = set()
        for x0, y0, x1, y1 in bbs:
            x0, y0, x1, y1 = (
                int(x0 * 255.0),
                int(y0 * 255.0),
                int(x1 * 255.0),
                int(y1 * 255.0),
            )
            if x0 < 0 or y0 < 0 or x1 < 0 or y1 < 0:
                continue
            else:
                corners_set.add((x0, y0))
                corners_set.add((x0, y1))
                corners_set.add((x1, y0))
                corners_set.add((x1, y1))
        cs_type_batch.append(list(corners_set))
    return cs_type_batch


def bb_to_seg(bbs_batch, im_size=256):
    """This functions seems to convert bbox to solid rectangles?"""

    all_rooms_batch = []
    for bbs in bbs_batch:
        areas = np.array([(x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in bbs])
        inds = np.argsort(areas)[::-1]
        bbs = bbs[inds]
        tag = 1
        rooms_im = np.zeros((256, 256))

        for (x0, y0, x1, y1) in bbs:
            if x0 < 0 or y0 < 0 or x1 < 0 or y1 < 0:
                continue
            else:
                room_im = Image.new("L", (256, 256))
                dr = ImageDraw.Draw(room_im)
                dr.rectangle(
                    (x0 * im_size, y0 * im_size, x1 * im_size, y1 * im_size),
                    outline="white",
                    fill="white",
                )
                inds = np.array(np.where(np.array(room_im) > 0))
                rooms_im[inds[1, :], inds[0, :]] = tag
                tag += 1

        all_rooms = []
        for tag in range(1, bbs.shape[0] + 1):
            room = np.zeros((256, 256))
            inds = np.array(np.where(rooms_im == tag))
            room[inds[0, :], inds[1, :]] = 1.0
            all_rooms.append(room)
        all_rooms_batch.append(all_rooms)
    all_rooms_batch = np.array(all_rooms_batch)

    return all_rooms_batch



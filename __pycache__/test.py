def get_class_from_id(idx):
    classes = []
    for id in idx:
        if id == 0:
            classes.append("cbb")
        elif id == 1:
            classes.append("cbsd")
        elif id == 2:
            classes.append("cgm")
        elif id == 3:
            classes.append("cmd")
        elif id == 4:
            classes.append("healthy")
    return classes

import os

TEMPLATE_DIRPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "templates"))


def get_template(filename):
    return os.path.join(TEMPLATE_DIRPATH, filename)

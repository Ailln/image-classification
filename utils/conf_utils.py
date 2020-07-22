import yaml


def get_conf(conf_path):
    with open(conf_path, "r") as f_conf:
        conf = yaml.load(f_conf, Loader=yaml.FullLoader)
    return conf

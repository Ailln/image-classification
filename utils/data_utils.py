import json


def save_json(data, save_path):
    with open(save_path, "w") as outfile:
        json.dump(data, outfile)


def read_json(read_path):
    with open(read_path, "r") as json_file:
        data = json.load(json_file)
        return data

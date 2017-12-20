import Algorithmia
from time import sleep
from requests.exceptions import ConnectionError
client = Algorithmia.client()


class AlgorithmError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)



def get_file(remote_file_path):
    try:
        result = client.file(remote_file_path).getFile().name
    except ConnectionError:
        result = get_file(remote_file_path)
    return result


def put_file(local_path, remote_path):
    try:
        client.file(remote_path).putFile(local_path)
    except ConnectionError:
        sleep(5)
        return put_file(local_path, remote_path)
    return remote_path

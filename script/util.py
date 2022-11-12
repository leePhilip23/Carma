import os

def midpoint(p1, p2):
    return [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2]
def read_api_key():
    """
    reads and parse and store the credential information from local sources
    :return:
    credd : dict
        a dictionary storing different keys and api keys
    """
    here_path = os.path.join(os.getcwd(), "..")
    here_path = os.path.join(here_path, "credentials")
    cred_path = os.path.join(here_path, "credentials.properties")
    # print(cred_path)
    cred = {}
    with open(cred_path, "r") as f:
        for temp in f:
            temp = (temp.strip()).split("=")
            # print(temp)
            cred[temp[0].strip()] = temp[1].strip()

    return cred
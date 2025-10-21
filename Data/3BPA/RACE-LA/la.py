from bam_torch.utils.utils import find_input_json, date
from bam_torch.laplace.post_hoc_laplace import PostHocLaplace
import json
import torch

if __name__ == '__main__':
    print(date())
    input_json_path = find_input_json()
    torch.cuda.empty_cache()

    with open(input_json_path) as f:
        json_data = json.load(f)

        approximator = PostHocLaplace(json_data)
        dict_key_y = 'energy' # 'energy' or 'forces_x' or 'forces_y' or 'forces_z'
        approximator.laplace_approximate(dict_key_y)

    print(date())


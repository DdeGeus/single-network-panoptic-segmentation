import json

def load_json_to_params(params, json_path=''):
  if json_path == '':
    pass
  else:
    with open(json_path, 'r') as fp:
      for k, v in json.load(fp).items():
        setattr(params, k, v)

  return params
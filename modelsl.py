import timm

if __name__ == "__main__":
  chosen_name = "vgg"


  for mode_name in timm.list_models(pretrained=True):
    if chosen_name in mode_name:
      print(mode_name)
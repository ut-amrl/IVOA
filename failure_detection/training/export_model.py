import sys, os, argparse
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from skimage import io
import torch
from collections import OrderedDict
from torchvision import transforms
import numpy as np
import cv2
from networks.failure_detection_multi_class import FailureDetectionMultiClassNet

# This script converts the trained pytorch model to a torch script via
# tracing or annotation

# Convert using tracing or annotation
USE_TRACING=True
LOAD_EXAMPLE_INPUT_FROM_FILE=False

def main():
  parser = argparse.ArgumentParser(description='Convert Pytorch models '
                                    'to Torch Script for use in CPP.')
  parser.add_argument(
                      "--input_model",
                      default=None,
                      help="Path to the trained pytorch model.",
                      required=True,
                      type=str)
  parser.add_argument("--output_model",
                    default=None,
                    help="Path to save the converted torch script model.",
                    required=True,
                    type=str)
  args = parser.parse_args()


  # Load the model
  net = FailureDetectionMultiClassNet()
  
  ## If the model has been saved using nn.DataParallel, the names
  ## of the keys of the model dictionary start with an extra 'module.'
  ## which should be removed
    
  # Map to CPU as you load the model
  state_dict = torch.load(args.input_model, 
                map_location=lambda storage, loc: storage)
  was_trained_with_multi_gpu = False
  new_state_dict = OrderedDict()
  for k, v in state_dict.items():
      if k.startswith('module'):
          was_trained_with_multi_gpu = True
          name = k[7:] # remove 'module.'
          new_state_dict[name] = v
          print(name)
  
  if(was_trained_with_multi_gpu):
      net.load_state_dict(new_state_dict)
  else:
      net.load_state_dict(state_dict)

  # *******************
  # Example image input:
  if LOAD_EXAMPLE_INPUT_FROM_FILE:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    data_transform_input = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    example_img = "/media/ssd2/datasets/Jackal_Visual_Odom/sequences/00037/image_0/000984.png"

    img = io.imread(example_img)
    if len(img.shape) > 2:
      img = img[:, : , 0:3]
      img = img.reshape((img.shape[0], img.shape[1], 3))
    else:
      img = img.reshape((img.shape[0], img.shape[1], 1))
      img = np.repeat(img, 3, axis=2)
    

    img = transforms.ToPILImage()(img)
    img = data_transform_input(img)
    img = img.reshape((1, 3, img.shape[1], img.shape[2]))
  else:
    #TODO: update width and height values
    img = torch.rand(1, 3, 600, 960)
  # ***********************

  # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
  if USE_TRACING:
    net.eval()
    script_module = torch.jit.trace(net, img)
    script_module.save(args.output_model)
  else:
    # Convert the pytorch model to a ScriptModule via annotation
    script_module = torch.jit.script(net)
    script_module.save(args.output_model)

  

  # ************************************
  # Test with an example image
  # ************************************
  script_module.eval()
  net.eval()

  for model, name in zip([script_module, net], ['scrpt_model', 'orig_model']):
    
    print('input img: ', img.shape)
    with torch.set_grad_enabled(False):
      output_img = model(img)
    print(type(output_img))
    print("output_img: ", output_img.shape)


    # Visualize input img
    input_img_np = img.numpy()
    input_img_np = input_img_np[0, 0, :, :]
    # Unnormalize the R channel of the input image:
    input_img_np = (input_img_np * 0.229) + 0.485
    input_img_cv8u = (255.0 * input_img_np).astype(np.uint8)
    input_img_cv8uc3 = cv2.cvtColor(input_img_cv8u, cv2.COLOR_GRAY2BGRA)
    cv2.imwrite("example_input.png", input_img_cv8uc3)

    # ***************************
    # Visualize output img
    # ***************************
    output_np = output_img.numpy()
    output_np = (255.0 * output_np).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(output_np, cv2.COLORMAP_JET)
    cv2.imwrite("output_" + name + ".png", heatmap_color)



if __name__=="__main__":
  main()
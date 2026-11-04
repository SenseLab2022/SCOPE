# from torchvision.models import resnet18
import torch
import pickle
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import confusion_matrix
import pdb
from datetime import datetime

from model.modules import ir_1w8a,binaryfunction
from model.resnet import resnet16
from general import parameters
from general.parameters import *
from general.adc_noise import *
from general.logger import log_print
from model.modules.replace_quant import *
torch.set_printoptions(profile="full")
torch.set_printoptions(precision=7, sci_mode=False)

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

def print_fx_node_shapes(gm: torch.fx.GraphModule, input_sample):
    print("==== Running Shape Trace ====")
    from torch.fx.passes.shape_prop import ShapeProp
    ShapeProp(gm).propagate(input_sample)

    for node in gm.graph.nodes:
        if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
            shape = node.meta['tensor_meta'].shape
            print(f"[{node.name}] {node.op} → {node.target}, shape: {shape}")
        else:
            print(f"[{node.name}] {node.op} → {node.target}")

def accuracy_inference(model, dataset, streamlit = False ):
    log_print("-------------------Inference Accuracy -----------------------")


    testXtr = unpickle(dataset)

    num_test = len(testXtr['labels'])
    y_out = np.zeros(num_test)
    y_label = np.zeros(num_test)
    for i in range(0, num_test):
        img = np.reshape(testXtr['data'][i], (3, 32, 32))
        img = img.transpose(1, 2, 0)
        img_show = img.astype(np.uint8)

        img_data = transforms.ToTensor()(img)
        img_data = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_data)
        img_data = torch.reshape(img_data, (1,3,32,32)).to("cuda")

        with torch.no_grad():

            output = model(img_data)
          
        y_label[i] = testXtr['labels'][i]
        y_out[i] = np.argmax(output.cpu().numpy(), axis = 1)



        if i %1 == 0 :
            print(str(datetime.now().strftime('%H:%M:%S')) +", "+str(i) + '_' + str(testXtr['labels'][i]) + '\tpredicted class:', y_out[i]) 
        cm = confusion_matrix(y_label, y_out)

    sum_correct = 0
    for k in range(10):
        sum_correct += cm[k, k]
    log_print('sum_correct: ',sum_correct)
    log_print('Accuracy(%):', sum_correct/num_test)



if __name__ == "__main__":


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet16().to(device)
    state_dict = torch.load("model/resnet_44000.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    if INFERENCE:
        model = replace_forward_function(model) 
        mode = replace_conv_with_irconv(model, ir_1w8a.IRConv2d_cim)
        print("replaced model with IRConv2d_CIM")
    else:
        model = replace_conv_with_irconv(model, ir_1w8a.IRConv2d)

    model = replace_forward(model, target_performance=0)
    
       
    variation_list = generate_variation()
    parastic_list = generate_parastic()
    P.cap_variation_tensor = torch.tensor(variation_list).to(device)
    P.cap_parasitic_tensor = torch.tensor(parastic_list).to(device)

    testXtr = unpickle("dataset/test_batch")
    num_test = 1000
    y_out = np.zeros(num_test)
    y_label = np.zeros(num_test)
    for i in range(0, num_test):
        img = np.reshape(testXtr['data'][i], (3, 32, 32))
        img = img.transpose(1, 2, 0)
        img_data = transforms.ToTensor()(img)
        img_data = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_data)
        img_data = torch.reshape(img_data, (1,3,32,32)).to("cuda")

        with torch.no_grad():
            # print_fx_node_shapes(model, img_data) 
            output = model(img_data)
          
        y_label[i] = testXtr['labels'][i]
        y_out[i] = np.argmax(output.cpu().numpy(), axis = 1)

        if i %10 == 0 :
            print(str(datetime.now().strftime('%H:%M:%S')) +", "+str(i) + '_' + str(testXtr['labels'][i]) + '\tpredicted class:', y_out[i]) 
        cm = confusion_matrix(y_label, y_out)
    print(cm)   
    sum_correct = 0
    for k in range(10):
        sum_correct += cm[k, k]
    print('sum_correct: ',sum_correct)


    



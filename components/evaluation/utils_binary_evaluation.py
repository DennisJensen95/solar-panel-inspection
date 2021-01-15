import torch
import time

def print_results(success_percent, fault_correct, no_fault_correct, fault_images, no_fault_images, num_images):
    
    print("---------------------------------------------")
    print(f'Succes percentage is: {success_percent}')
    print(f'Faults succesfully found: {fault_correct/fault_images}')
    print(f'No faults succesfully not found: {no_fault_correct/no_fault_images}')
    print(f'Total images: {num_images}')
    print(f'Images with faults: {fault_images}')
    print(f'Images without fault: {no_fault_images}')
    print("---------------------------------------------")

@torch.no_grad()
def evaluate_binary(model, data_loader_test, device, prediction_certainty_cutoff):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")

    # Put in evaluation mode
    model.eval()

    data_iter_test = iter(data_loader_test)
    # iterate over test subjects
    success = 0
    num_images = 0
    fault_correct = 0
    no_fault_correct = 0
    fault_images = 0
    no_fault_images = 0
    for images, label in data_iter_test:
        images = list(img.to(device) for img in images)
        label = label[0]

        # torch.cuda.synchronize()  # what is this??
        model_time = time.time()
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time
        
        score = outputs[0]["scores"].numpy()
        
        # Check if model thinks there is a fault
        if len(score) > 0:
            if score[0] > prediction_certainty_cutoff:
                label_pred = 1
            else:
                label_pred = 0
        else:
            label_pred = 0
        
        # Correct label succes
        if label == label_pred:
            success += 1
        
        # Check which correct labelling if correct
        if label == 0 and label == label_pred:
            no_fault_correct += 1
        elif label == 1 and label == label_pred:
            fault_correct += 1
        
        # No fault images
        if label == 0:
            no_fault_images += 1
        
        # Fault images
        if label == 1:
            fault_images += 1
        
        num_images += 1
        success_percent = success / num_images
        
        if num_images % 100 == 0:    
            print_results(success_percent, fault_correct, no_fault_correct, fault_images, no_fault_images, num_images)
    

    
    
    torch.set_num_threads(n_threads)

    # Set back in training mode
    model.train()

    return success_percent, fault_correct, no_fault_correct, fault_images, no_fault_images, num_images

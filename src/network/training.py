
import matplotlib.pyplot as plt
import numpy as np
import torch as torch

def validate(dataset, model, accuracy, loss_function, primary_info, secondary_info):
    mse = 0
   # model.eval()
    for i, (training_data, target) in enumerate(dataset):
            #model.eval()
            output = model(training_data)
            error = abs(target[0].item() - output[0].item()) / target[0].item() * 100
            if secondary_info: 
                print(f'Target: {target[0].item() * 1000} | Output: {output[0].item() * 1000} | Error: {error}%')
            mse += error / len(dataset)
    if primary_info:
        print(f"Performance: {mse}%, validation size: {len(dataset)}")
    accuracy.append(mse)

def train(model, dataset_train, dataset_validate, loss_function, optimizer, step, losses, axs, plot_data, stop_condition = 0.0006, epochs = 100, validation_range = 10, visualization = False, secondary_info = False, primary_info = True, model_save = False):
    flag = False
    accuracy = []
    step_val =[]
    if visualization:
        axs[0].set_title('Loss Training Set')
        axs[1].set_title('Average Error Validation Set')
        axs[0].set_xlabel('Epochs')
        axs[1].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[1].set_ylabel('Average Error %')
        axs[1].set_ylim(0, 15)
        
        plt.ion()
        plt.show()
    for epoch in range(epochs):
        if flag:
            break
        model.train()
        mse = 0
        for i, (training_data, target) in enumerate(dataset_validate):
            optimizer.zero_grad()
            output = model(training_data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            mse += loss.item()
        step.append(epoch)
        losses.append(mse)
        axs[0].plot(step, losses)
        if visualization:
            plt.draw()
            plt.pause(0.1)
        # if mse < stop_condition:
        #     flag = True
        if epoch % validation_range == 0 :
            validate(dataset_validate, model, accuracy,  loss_function, primary_info=primary_info,secondary_info=secondary_info)
            if model_save:
                torch.save(model.state_dict(), "model_v_0_1.pth") 
            step_val.append(epoch)
            if visualization: axs[1].plot(step_val, accuracy)
        if secondary_info :print(f'Epoch {epoch}, Batch {i}, Loss: {mse}')
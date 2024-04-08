import matplotlib.pyplot as plt


def predict_error_visual(batch_grads, batch_params, batch_importants, output, save_path):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    batch_grads = batch_grads.cpu().numpy()
    batch_params = batch_params.cpu().numpy()
    batch_importants = batch_importants.cpu().numpy()
    output = output.cpu().numpy()
    ax.scatter(batch_params, batch_grads, batch_importants, c='r', marker='o')
    ax.scatter(batch_params, batch_grads, output, c='b', marker='o')
    
    plt.title('True Values vs Predictions')
    plt.savefig(save_path + '_predict.png')
    # plt.show()
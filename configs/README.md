## Mandatory config elements
| Config Entry                          | Type         | Description                                    | Mandatory |
| :---                                  | :---:        |      :---:                                     |  :---:    |
| .experiment.name                      | String       | Name of the current experiment.                |   ✅   |
| .experiment.use_cuda                  | Boolean      | True if GPU shall be used.                     |   ✅   |
| .agent.name                           | String       | Name of the Subclass Agent.                    |   ✅   |
| .agent.use_tensorboard                | Boolean      | True if TensorBoard shall be used for logging. |   ❌   |
| .agent.use_wandb                      | Boolean      | True if WandB shall be used for logging.       |   ❌   |
| .agent.num_classes                    | int          | Number of classes to predict.                  |   ❌   |
| .agent.max_epoch                      | int          | Number of epochs to train.                     |   ✅   |
| .model_wrapper.name                   | String       | Name of the Subclass ModelWrapper.             |   ✅   |
| .model_wrapper.network.name           | String       | Name of the Subclass Network Architecture.     |   ✅   |
| .model_wrapper.network.input_channels | int          | Number of channels of the input tensor.        |   ❌   |
| .dataset.name                         | String       | Name of the Subclass DataSet.                  |   ✅   |
| .dataset.root                         | String       | Root path for the DataSet to be initialized.   |   ✅   |
| .dataset.ignore_mask_value            | int          | Index which should be ignored while training.  |   ✅   |
| .dataset.crop_images                  | Boolean      | True if images shall be cropped.               |   ❌   |
| .dataset.data_augmentation            | Boolean      | True if data augmentation shall be processed.  |   ❌   |
| .dataloader.validation_ratio          | float [0, 1] | Relative part of the validation part.          |   ✅   |
| .dataloader.batch_size                | int          | Batch size of the training dataloader.         |   ✅   |
| .dataloader.num_workers               | int          | Number of workers to be used.                  |   ✅   |
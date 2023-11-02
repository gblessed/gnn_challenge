# gnn_challenge

torch_routnet.ipynb - implementation of routenet using pytorch

extracted_08_10 -  test_data
extracted_08_10_cv  - training data

SWS_ITU.docx - report document


training_history.txt - training and validation loss history during each epoch

To make predictions using the trained model use: 


python predict.py -ds CBR+MB --ckpt-path 22-41.9264 --tr-path "extracted_08_10_cv\0\training" --te-path "extracted_08_10"




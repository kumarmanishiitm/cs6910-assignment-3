Assignment 3 submitted by Manish Kumar (cs21m033) and Vishal Yadav (cs21m073)


************* Executing through Command Line ************************
1. import q1_final.py file into your colab notebook.
2. import the dataset into your colab notebook.
3. Run the below command to execute the file
	!python <filename> <type_layer> <encoder_layer> <decoder_layer> <units> <dropout> <attention> <embedding_dim> <beam_search>	
	eg: !python q1_final.py lstm 3 3 256 0.11 True 256 False

*************************************************************************





1. Run the notebook cell by cell.
2. To run model on test data the followin code executed
		model = Test_Model(lang="hi",embed_dim=256,enc_layers=3,dec_layers=3,type_layer="lstm",units=256,dropout=0.2,attention=True)                                                                                                                                                          

3. To run the model with WandB sweep, use the following code:
```python
# Creating the WandB config
sweep_config = {
  "name": "Sweep_Assignment3",
  "method": "random",
  "parameters": {
        "decorder_encoder_layers": {
           "values": [1, 2, 3]
        },
        "units": {
            "values": [64, 128, 256]
        },
        "type_of_Layer": {
            "values": ["gru", "rnn","lstm"]
        },
         "embeded_dim": {
            "values": [256,64, 128]
        },
        "dropout": {
            "values": [0.29, 0.37]
        },
        "Beam_width": {
            "values": [3, 7, 5]
        },
        "Teacher_forcing_ratio": {
            "values": [0.9, 0.5,0.2]
        },
        "Attention": {
            "values": [True,False]
        },
        "epochs":{
            "values":[10,20,30]
        }

    }
}
```
4. To visualize the model outputs, use the following code:
```python
visualize_model_outputs(model, n=15)
```
5. To visualise the model connectivity, use the following code:
```python
# Sample some words from the test data
test_words = get_test_words(5)
# Visualise connectivity for "test_words"
for word in test_words:
    visualise_connectivity(model, word, activation="scaler")
```
6. Question 8 is at the bottom of the notebook. To execute the code for question 8, run the code cell by cell.


# Detect Many Body Localization

Python code that generates eigenvectors / reduced density matrices of a 1D Heisenberg spin chain with a random field, using exact diagonalization. 
Eigenstates generated from disorder strengths _W_ = 0.5 and _W_ = 8.0 are stored as training data for a PyTorch CNN. 
The critical disorder strength _W<sub>c</sub>_ of the phase transition is determined by using the CNN to detect to classify samples with random _W_.

The process is orchestrated by three sets of Jupyter Notebooks:
* Data Generation
* Neural Network Training
* Result Plotting

To facilitate mounting Google Drive on Google Colab, the content should be copied to a new Notebook native to Colab.
First upload the Notebook to Colab, then Ctrl+Shift+A (select all cells) and then Ctrl+C (copy).
Then create a new Notebook on Colab, and Ctrl+V (paste) the content.
View the file directory on the left sidebar, which should automatically allocate an instance, and connect Google Drive by click the Drive logo.
Google Drive should be able to connect without having to run any code and copy-and-paste passcode strings.
`.py` codes should be uploaded to the appropriate Google Drive directory to avoid needing to re-upload them every time.

This code is originally used in an university course project. 
As such, the techniques utilized are rudimentary. 
System size _L_ is thus limited to 12.
If higher _L_ is desired, the exact diagonal step should be accelerated using symmetries in the Hamiltonian, or replaced with an evolution of Matrix Product State (MPS).



# Results

![Reduced density matrices as images](https://github.com/tinkei/detect-many-body-localization/blob/master/resources/reduced-density_L12-n6-periodic-k5.png?raw=true)

Fig. 1.
Typical magnitudes of reduced density matrices at _W_ = 0.5 (left columns) and _W_ = 8.0 (right columns). 
Generated using parameters _L_ = 12, _n_ = 6, _k_ = 5, _periodic_.

![CNN classifer](https://github.com/tinkei/detect-many-body-localization/blob/master/resources/model-prediction_L12-n6-periodic-k5.png?raw=true)

Fig. 2.
Visualizing CNN predictions for random _W_, with a few artificial "failure cases".
Generated using parameters _L_ = 12, _n_ = 6, _k_ = 5, _periodic_.

![Sigmoid curve fitting to find critical disorder strength](https://github.com/tinkei/detect-many-body-localization/blob/master/resources/curve-fitting_L10-n6-periodic-k1.png?raw=true)

Fig. 3.
Probabilities of being in a localized phase for random _W_, as predicted by CNN. 
The raw probabilities are plotted in blue dots. 
Their average over each random _W_ are shown as blue crosses. 
Data is fitted with a sigmoid curve (orange dashed line), from which the critical disorder strength _W<sub>c</sub>_ is found to be 2.6426 &pm; 0.0025.
Generated using parameters _L_ = 10, _n_ = 6, _k_ = 1, _periodic_.

![Results (critical disorder strength)](https://github.com/tinkei/detect-many-body-localization/blob/master/resources/scaling_W_dataset3.png?raw=true)

Fig. 4.
Scaling behavior of critical disorder strength _W<sub>c</sub>_. 
_W<sub>c</sub>_ is found to be around 2.5 - 3.0 for larger _L_ and _n_.

![Results (steepness of transition)](https://github.com/tinkei/detect-many-body-localization/blob/master/resources/scaling_b_dataset3.png?raw=true)

Fig. 5.
Scaling behavior of transition steepness _b_ (i.e. width of transition region). 

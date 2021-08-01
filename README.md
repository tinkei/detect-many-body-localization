# Detect Many Body Localization

Many Body Localization (MBL) is a phenomenon whereby isolated many body systems do not thermalize, against the predictions of statistical mechanics. 
They exhibit a phase transition depending on a disorder strength _W_. 

By considering reduced density matrices of infinite-temperature eigenstates (_E_ &#x2248; 0) as images, 
as generated from a 1D Heisenberg spin chain with a random field, 
this code uses a classifier neural network's output probability to fit a critical disorder strength _W<sub>c</sub>_ of the phase transition. 
Eigenstates generated from disorder strengths _W_ = 0.5 and _W_ = 8.0 are stored as training data for a PyTorch CNN. 
The CNN is used to classify samples generated from random _W_, 
where the classifcation probability is fitted with a sigmoid function to determine _W<sub>c</sub>_.
Various scaling with _W<sub>c</sub>_ are illustrated.

The process is orchestrated by three sets of Jupyter Notebooks:
* Data Generation
* Neural Network Training
* Result Plotting

When using Google Colab, `.py` codes should be uploaded to the appropriate Google Drive directory to avoid needing to re-upload them every time.

Due to the difficulty in restarting long-running computations, a `shutdown_signal.txt` with content `1` could be placed at the root to gracefully exit a for-loop only when a full iteration is completed.

To facilitate mounting Google Drive on Google Colab, the content should be copied to a new Notebook native to Colab.
First upload the Notebook to Colab, then Ctrl+Shift+A (select all cells) and then Ctrl+C (copy).
Then create a new Notebook on Colab, and Ctrl+V (paste) the content.
View the file directory on the left sidebar, which should automatically allocate an instance, and connect Google Drive by click the Drive logo.
Google Drive should be able to connect without having to run any code and copy-and-paste passcode strings.

This code was originally used in an university course project. 
As such, the techniques utilized are rudimentary. 
System size _L_ is thus practically limited to 12.
If higher _L_ is desired, the exact diagonal step should be accelerated using symmetries in the Hamiltonian, or replaced with an evolution of Matrix Product State (MPS).



## Tasks (v1.X.X):

- [x] Publish original source code.
- [ ] Use environment variables os.environ['running_on_colab'] instead of a bool.
- [ ] Refactor `util` functions.
- [ ] Read CNN hyperparameters from `.json` file.

## Tasks (v2.X.X):

- [ ] Store training data as eigenvectors and do partial trace in runtime.

## Tasks (v3.X.X):

- [ ] Advanced methods to compute eigenstates.
- [ ] Implement "Learning phase transitions by confusion" (van Nieuwenburg et al., 2017)



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

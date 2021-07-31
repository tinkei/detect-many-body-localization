# Detect Many Body Localization

Python code that generates eigenvectors / reduced density matrices of a 1D Heisenberg spin chain with a random field, using exact diagonalization. 
Eigenstates generated from disorder strengths _W_ = 0.5 and _W_ = 8.0 are stored as training data for a PyTorch CNN. 
The critical transition disorder strength is determined by using the CNN to detect to classify samples with random _W_.

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

This code is originally used in an university course project. 
As such, the techniques utilized are rudimentary. 
System size _L_ is thus limited to 12.
If higher _L_ is desired, the exact diagonal step should be accelerated using symmetries in the Hamiltonian, or replaced with an evolution of Matrix Product State (MPS).

---

![Reduced density matrices as images](https://github.com/tinkei/detect-many-body-localization/blob/master/resources/reduced-density_L12-n6-periodic-k5.png?raw=true)

![CNN classifer](https://github.com/tinkei/detect-many-body-localization/blob/master/resources/model-prediction_L12-n6-periodic-k5.png?raw=true)

![Sigmoid curve fitting to find critical disorder strength](https://github.com/tinkei/detect-many-body-localization/blob/master/resources/curve-fitting_L10-n6-periodic-k1.png?raw=true)

![Results (critical disorder strength)](https://github.com/tinkei/detect-many-body-localization/blob/master/resources/scaling_W_dataset3.png?raw=true)

![Results (steepness of transition)](https://github.com/tinkei/detect-many-body-localization/blob/master/resources/scaling_b_dataset3.png?raw=true)

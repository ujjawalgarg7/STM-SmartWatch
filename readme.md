# 🚀 STM - Smart Watch

## 📌 Overview  
This project includes a machine learning model for training and testing, along with additional utilities for handling location-based data and email notifications.  

## ⚙️ Installation  

### 1⃣ Clone the Repository  
```bash
git clone https://github.com/ujjawalgarg7/STM-SmartWatch
cd STM-SmartWatch
```

### 2⃣ Set Up a Virtual Environment  
Create a virtual environment to keep dependencies isolated:  
```bash
python -m venv venv
```

Activate the virtual environment:  
- **On Windows (Command Prompt)**  
  ```bash
  venv\Scripts\activate
  ```
- **On Windows (PowerShell)**  
  ```powershell
  venv\Scripts\Activate.ps1
  ```
- **On macOS/Linux**  
  ```bash
  source venv/bin/activate
  ```

### 3⃣ Install Dependencies  
Once the virtual environment is activated, install the required packages:  
```bash
pip install -r requirements.txt
```

---

## 📂 Dataset  

### 👅 Download Training and Testing Data  
To train and test the model, download the datasets:  

- **💜 Training Set:** [Download here](<https://drive.google.com/drive/folders/1zgQ8bCxi7Vu5CAe5nLPFCIIYGHEF8mKz?usp=drive_link>)  
- **💜 Testing Set:** [Download here](<https://drive.google.com/drive/folders/1GNyrwcUovqKaJI0rAvbtFfeQXYgGqHde?usp=drive_link>)  

After downloading, place the datasets in the appropriate directories:

```bash
mkdir Dataset Testing
mv <downloaded-training-file> Dataset/
mv <downloaded-testing-file> Testing/
```

---

## 🚀 Usage  

### 🏋️ Train the Model  
```bash
python ModelTrain.py
```

### 🧪 Test the Model  
```bash
python ModelTest.py
```

### 📍 Location Handling  
```bash
python location.py
```

### ✉️ Send Email Notifications  
```bash
python mail.py
```

---

## 📂 File Structure  

📎 **Project Folder**  
📄 `.gitignore` - Specifies files to be ignored by Git.  
📄 `ModelTrain.py` - Trains the machine learning model.  
📄 `ModelTest.py` - Tests the trained model on new data.  
📄 `location.py` - Handles location-based functionalities.  
📄 `mail.py` - Manages email notifications.  
📄 `requirements.txt` - Contains dependencies required for the project.  
📂 `data/` - Directory for training and testing datasets.  

---

## ❌ Deactivating the Virtual Environment  
Once you're done working, deactivate the virtual environment:  
```bash
deactivate
```

---

## 💡 Contributing  
Contributions are welcome! If you'd like to contribute, please fork the repository and submit a pull request.  

## 📝 License  
This project is licensed under the MIT License.  

---

### ✨ Happy Coding! 🚀🔥  


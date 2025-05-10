# 💜 Code Purple

**Code Purple** is a web-based tool that scans policy documents for hidden gender bias and helps policymakers craft more inclusive, equitable policies—not just on paper, but in practice.

## 🚀 Why Code Purple?

The world is shifting. As we push for progress, unconscious and systemic biases still quietly shape decision-making spaces. **Code Purple** exists to uncover those subtle patterns, enabling leaders to build policies that actually work for everyone.

## 🧠 What It Does

- Scans policy or legal documents
- Detects potential gender bias using a fine-tuned BERT model
- Displays bias scores and highlights biased sections
- Suggests areas to rethink for more inclusive policymaking

## 🛠 Tech Stack

- 🤖 [BERT](https://huggingface.co/) (pre-trained transformer model, fine-tuned on our dataset)
- 🐍 Python
- 📊 Streamlit (for the frontend interface)
- 📁 Pandas, NumPy (for data handling)
- ✨ Custom dataset creation for bias detection

## 🌐 Try It Live

👉 [Link to the App](https://codepurple.streamlit.app/) 

## 📦 How to Run Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/code-purple.git
   cd code-purple
   pip install -r requirements.txt
   python train_model.py
   streamlit run app.py
   ```

## 📚 Our Learning Journey

We discovered that:

- Bias is rarely obvious—it often hides in structure, tone, or omission.  
- Even the most neutral-sounding policies can disadvantage certain groups.  
- Creating a balanced training dataset was tough but essential.  

## ⚠️ Challenges We Faced

- Finding real-world policy documents  
- Creating and labeling unbiased vs biased examples  
- Ensuring the model didn’t overfit or misinterpret language nuance  

## ✨ Vision

- Help create more policies that are gender inclusive
- Get more datasets so that we can train the model better and make this initiative even more successful

---

*Built with purpose and caffeine by a team who believes that inclusive futures begin with inclusive policies.*

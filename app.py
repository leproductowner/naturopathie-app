import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Charger le modèle Qwen
model_name = "Qwen/Qwen-7B"  # Remplacez par le modèle Qwen que vous souhaitez utiliser
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Fonction pour générer le bilan naturopathique
def generer_bilan(sexe, age, etat_civil, occupation, stress_travail, 
                  regime, eau, sucre, sommeil, reveils, 
                  activite_physique, type_activite, jambes_lourdes, tabac, 
                  pollution, respiration, ant_med_perso, ant_med_fam, 
                  digestion, fibres, gestion_stress, satisfaction, raison_consultation):
    
    # Création du prompt détaillé
    prompt = f"""
    Agis en tant que naturopathe expert pour établir un bilan naturopathique détaillé en utilisant les informations suivantes :
    
    - **Sexe** : {sexe}
    - **Âge** : {age}
    - **État civil** : {etat_civil}
    - **Occupation** : {occupation}
    - **Niveau de stress au travail** : {stress_travail}
    - **Régime alimentaire** : {regime}
    - **Consommation d'eau quotidienne** : {eau}
    - **Consommation de sucre** : {sucre}
    - **Heures de sommeil** : {sommeil}
    - **Réveils nocturnes** : {reveils}
    - **Activité physique** : {activite_physique}
    - **Type d'activité physique** : {type_activite}
    - **Sensation de jambes lourdes** : {jambes_lourdes}
    - **Tabagisme** : {tabac}
    - **Exposition à la pollution** : {pollution}
    - **Pratique de respiration (ex. méditation)** : {respiration}
    - **Antécédents médicaux personnels** : {ant_med_perso}
    - **Antécédents médicaux familiaux** : {ant_med_fam}
    - **Problèmes digestifs** : {digestion}
    - **Consommation de fibres** : {fibres}
    - **Gestion du stress** : {gestion_stress}
    - **Satisfaction globale dans la vie** : {satisfaction}
    - **Raison de la consultation** : {raison_consultation}
    
    Fournis des recommandations détaillées couvrant les catégories suivantes :
    
    1️⃣ **Alimentation** 🍏  
    2️⃣ **Sommeil** 😴  
    3️⃣ **Activité physique** 🏃‍♂️  
    4️⃣ **Gestion du stress** 🌿  
    5️⃣ **Respiration** 💨  
    6️⃣ **Autres aspects pertinents pour la santé globale** 🧘‍♂️  
    """

    # Générer le texte avec Qwen
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=500, do_sample=True, temperature=0.7)
    bilan_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Générer le PDF
    pdf_filename = "bilan_naturopathique.pdf"
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    c.setFont("Helvetica", 12)
    
    y_position = 750
    for line in bilan_text.split("\n"):
        c.drawString(50, y_position, line)
        y_position -= 20
        if y_position < 50:
            c.showPage()
            y_position = 750

    c.save()
    
    return bilan_text, pdf_filename

# Interface Streamlit
st.title("Bilan Naturopathique Personnalisé")
st.write("Remplissez ce formulaire et recevez un bilan naturopathique détaillé avec un fichier PDF téléchargeable.")

with st.form("bilan_form"):
    sexe = st.radio("Sexe", ["Homme", "Femme", "Autre"])
    age = st.radio("Âge", ["Moins de 18 ans", "18-30 ans", "31-50 ans", "Plus de 50 ans"])
    etat_civil = st.radio("État civil", ["Célibataire", "Marié(e)", "Divorcé(e)", "Veuf/Veuve"])
    occupation = st.text_input("Occupation")
    stress_travail = st.radio("Niveau de stress au travail", ["Faible", "Modéré", "Élevé"])
    regime = st.radio("Régime alimentaire", ["Omnivore", "Végétarien", "Végétalien", "Autre"])
    eau = st.radio("Consommation d'eau quotidienne", ["Moins de 1 litre", "1-2 litres", "Plus de 2 litres"])
    digestion = st.text_input("Problèmes de digestion")
    raison_consultation = st.text_input("Raison de la consultation")

    submitted = st.form_submit_button("Générer le bilan")
    if submitted:
        bilan_text, pdf_filename = generer_bilan(
            sexe, age, etat_civil, occupation, stress_travail, 
            regime, eau, "", "", "", "", "", "", "", 
            "", "", digestion, "", "", "", raison_consultation
        )
        st.write("### Résumé du bilan")
        st.write(bilan_text)
        with open(pdf_filename, "rb") as pdf_file:
            st.download_button("Télécharger le bilan PDF", pdf_file, file_name=pdf_filename)

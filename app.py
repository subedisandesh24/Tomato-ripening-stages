# ---------------- LEAF DISEASE CLASSIFIER ----------------
with tab3:
    disease_file = st.file_uploader("Upload a tomato leaf image", type=["jpg", "png", "jpeg", "heic"])
    if disease_file:
        if disease_file.type == "image/heic":
            heif_file = pillow_heif.read_heif(disease_file.read())
            disease_img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data)
        else:
            disease_img = Image.open(disease_file)

        st.image(disease_img, caption="Uploaded Leaf Image", use_column_width=True)

        disease_results = disease_model(disease_img)
        probs = disease_results[0].probs

        # Top-3 predictions
        top3_indices = probs.top5[:3]
        st.subheader("Top-3 Disease Predictions 🌿")
        for idx in top3_indices:
            class_name = disease_model.names[idx]
            confidence = probs.data[idx]
            if idx == probs.top1:
                st.markdown(f"- 🔴 **{class_name}** → Confidence: `{confidence:.2f}`")
            else:
                st.write(f"- {class_name}: {confidence:.2f}")

        # Show management strategy for top-1 disease only
        major_class = disease_model.names[probs.top1].lower()

        st.subheader("Recommended Management Strategy 🌿")

        if "bacterial spot" in major_class:
            st.write("""
**Chemical:** Copper Oxychloride 50% WP  
**Brands (Nepal):** Blitox, Blue Copper, Cu-50  
**Dosage:** 2–3 g per liter of water  
**Note:** Spray early morning or late evening to avoid leaf burn
            """)

        elif "early blight" in major_class or "late blight" in major_class:
            st.write("""
**Protective Chemical:** Mancozeb 75% WP  
**Brands:** Dithane M-45, Indofil M-45  
**Curative Chemical:** Metalaxyl 8% + Mancozeb 64% WP  
**Brands:** Krilaxyl, Ridomil Gold, Matco  
**Dosage:** 2 g per liter of water
            """)

        elif "leaf mold" in major_class:
            st.write("""
**Chemical:** Carbendazim 50% WP  
**Brands:** Bavistin, Beve-50  
**Dosage:** 1–2 g per liter of water  
**Alternative:** Chlorothalonil (Kavach)
            """)

        elif "powdery mildew" in major_class:
            st.write("""
**Chemical:** Wettable Sulphur 80% WP or Hexaconazole 5% EC  
**Brands:** Sulfex, Contaf, Sitara  
**Dosage:** 2 g per liter (Sulphur) or 2 ml per liter (Hexaconazole)
            """)

        elif "septoria" in major_class:
            st.write("""
**Chemical:** Chlorothalonil 75% WP  
**Brands:** Kavach, Ishan  
**Dosage:** 2 g per liter of water
            """)

        elif "spider mite" in major_class:
            st.write("""
**Chemical:** Abamectin 1.8% or 1.9% EC  
**Brands:** Vertimec, Abacin, V-mectin  
**Dosage:** 0.5–1 ml per liter of water  
**Note:** Spray underside of leaves where mites hide
            """)

        elif "target spot" in major_class:
            st.write("""
**Chemical:** Azoxystrobin 23% SC or Mancozeb  
**Brands:** Amistar, Mirador  
**Dosage:** 1 ml per liter of water
            """)

        elif "yellow leaf curl" in major_class or "tylcv" in major_class:
            st.write("""
**Disease Type:** Viral (TYLCV) — no chemical cure  
**Vector Control:** Whitefly (Bemisia tabaci)  
**Chemical:** Imidacloprid 17.8% SL or Acetamiprid 20% SP  
**Brands:** Confidor, Media, Pride, Manik  
**Dosage:** 0.5 ml (Imidacloprid) or 0.5 g (Acetamiprid) per liter of water
            """)

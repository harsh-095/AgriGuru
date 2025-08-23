## AgriGuru

Progress

1. Crop Suggestion, Images: 1. Crop detection 2. Crop disease detection

2. SQL Query Generation

Next To-Do

1. Query Expansion - (Check need for sql generation if not general,if yes combine with nrml msg)

2. Memory & Cache

3. Add audio conversion

4. For Image fetch Match Percent, and top 3 matches

Planned

1. Query Expansion - To have higher hit rate
2. Multi-Retrieval (1st level- ANN:approx nearest, 2nd level - Finer )
3. Memory For Resonse
4. Cache for responses
5. Context Expansion
6. Text, Image, Audio , Video Embeddings
7. Performance Optimization
8. Use OpenRoute or Mistral apis
9. Query based searchs , SQL query integration
   10 . Optimization Techniques like User Feedback
10. Invalidating wrong response in cache or remove old data
11. Combining all apis in one, crop , image

#### Set Up

Ref: https://github.com/whyashthakker/RAG

```
pip install streamlit requests
pip install python-multipart
pip install sentence-transformers torchvision pillow numpy
pip install fastapi uvicorn
pip install -U langchain-huggingface
pip install -U langchain-community
pip install faiss-cpu langchain sentence-transformers pandas langchain-ollama sqlite3 sqlalchemy langchain langchain-experimental mistralai
```

Files:
BE : full_be_api.py
UI : ui_fe.py

Used Models

```
For Image to Text Embedding: For Index Creation: clip-ViT-B-32

Crop Recommendation
For Embedding: sentence-transformers/all-MiniLM-L6-v2

General Model: gemma3:1b
```

Run Commands:

# UI

Run Using
streamlit run ui_fe.py

# BE

uvicorn full_be_api:app --reload --host 0.0.0.0 --port 8000

# Latest Improvements for SQL query based search

## Example 1

```
Question:what all crops data do you have
=========================================================
SQL:SELECT DISTINCT label FROM crops;
=========================================================
Result:[('rice',), ('maize',), ('chickpea',), ('kidneybeans',), ('pigeonpeas',), ('mothbeans',), ('mungbean',), ('blackgram',), ('lentil',), ('pomegranate',), ('banana',), ('mango',), ('grapes',), ('watermelon',), ('muskmelon',), ('apple',), ('orange',), ('papaya',), ('coconut',), ('cotton',), ('jute',), ('coffee',)]
=========================================================
Answer:Sure, I can help with that. Based on the data you have, I can provide the following crops information:

* You have data on various types of crops, including food crops such as rice, maize, chickpea, kidneybeans, pigeonpeas, mothbeans, mungbean, blackgram, lentil, pomegranate, banana, mango, grapes, watermelon, muskmelon, apple, orange, and papaya.
* In addition, you also have data on cash crops such as cotton and jute, as well as plantation crops like coconut.
* Finally, you also have data on coffee, which is a beverage crop.

Let me know if you have any more questions!
```

## Example 2

```
Question:Give report of all crops and ranges of the soil requirements
=========================================================
SQL:SELECT
    label AS crop,
    MIN(N) AS min_N,
    MAX(N) AS max_N,
    MIN(P) AS min_P,
    MAX(P) AS max_P,
    MIN(K) AS min_K,
    MAX(K) AS max_K,
    MIN(temperature) AS min_temperature,
    MAX(temperature) AS max_temperature,
    MIN(humidity) AS min_humidity,
    MAX(humidity) AS max_humidity,
    MIN(ph) AS min_ph,
    MAX(ph) AS max_ph,
    MIN(rainfall) AS min_rainfall,
    MAX(rainfall) AS max_rainfall
FROM
    crops
GROUP BY
    label;
=========================================================
Result:[('apple', 0, 40, 120, 145, 195, 205, 21.0365275, 23.99686172, 90.02575116, 94.92048112, 5.514253142, 6.4992268210000015, 100.1173443, 124.9831618), ('banana', 80, 120, 70, 95, 45, 55, 25.01018457, 29.90888522, 75.03193255, 84.97849241, 5.505393832999999, 6.490074429, 90.10978128, 119.84797), ('blackgram', 20, 60, 55, 80, 15, 25, 25.09737391, 34.9466155, 60.06534859, 69.96100028, 6.500144962, 7.775306272000001, 60.41790253, 74.91559514), ('chickpea', 20, 60, 55, 80, 75, 85, 17.02498456, 20.99502153, 14.25803981, 19.96978871, 5.988992796000002, 8.868741443, 65.11365631, 94.78189594), ('coconut', 0, 40, 5, 30, 25, 35, 25.00872392, 29.8690834, 90.01734526, 99.98187601, 5.50158009, 6.470465614, 131.09000759999998, 225.6323656), ('coffee', 80, 120, 15, 40, 25, 35, 23.05951896, 27.92374437, 50.04557009, 69.94807345, 6.020947179, 7.493191968, 115.1564012, 199.4735636), ('cotton', 100, 140, 35, 60, 15, 25, 22.00085141, 25.99237426, 75.00539324, 84.87668973, 5.801047545, 7.994679507000001, 60.65381719, 99.93100821), ('grapes', 0, 40, 120, 145, 195, 205, 8.825674745, 41.94865736, 80.01639435, 83.98351748, 5.510924848999999, 6.499604931, 65.01095312, 74.91506217), ('jute', 60, 100, 35, 60, 35, 45, 23.09433785, 26.98582182, 70.88259632, 89.89106506, 6.002524871, 7.4880144039999985, 150.2355238, 199.83629130000003), ('kidneybeans', 0, 40, 55, 80, 15, 25, 15.33042636, 24.92360104, 18.09224048, 24.96969858, 5.502999119, 5.99812453, 60.27552528, 149.7441028), ('lentil', 0, 40, 55, 80, 15, 25, 18.06486101, 29.94413861, 60.09116626, 69.92375891, 5.91645379, 7.841496029, 35.03484812, 54.93937710000001), ('maize', 60, 100, 35, 60, 15, 25, 18.04185513, 26.54986394, 55.28220433, 74.82913698, 5.513697923, 6.995843776, 60.65171481, 109.7515385), ('mango', 0, 40, 15, 40, 25, 35, 27.00315545, 35.99009679, 45.02236377, 54.9640534, 4.507523551, 6.9674177660000005, 89.29147581, 100.8124659), ('mothbeans', 0, 40, 35, 60, 15, 25, 24.01825377, 31.99928579, 40.00933429, 64.95585424, 3.504752314, 9.93509073, 30.92014047, 74.44330654), ('mungbean', 0, 40, 35, 60, 15, 25, 27.01470397, 29.914544300000006, 80.03499648, 89.99615558, 6.218923893, 7.199495367999999, 36.12042927, 59.87232071), ('muskmelon', 80, 120, 5, 30, 45, 55, 27.02415146, 29.94349168, 90.01506395, 94.96218673, 6.002927293, 6.781050372999999, 20.21126747, 29.86681385), ('orange', 0, 40, 5, 30, 5, 15, 10.01081312, 34.90665289, 90.00621688, 94.96419851, 6.010391864, 7.995848977, 100.1737964, 119.6946577), ('papaya', 31, 70, 46, 70, 45, 55, 23.012401800000006, 43.67549305, 90.03863107, 94.94482086, 6.501521192, 6.993473247000001, 40.35153141, 248.8592986), ('pigeonpeas', 0, 40, 55, 80, 15, 25, 18.31910448, 36.97794384, 30.40046769, 69.69141302, 4.548202098, 7.445444882999999, 90.05422663, 198.8298806), ('pomegranate', 0, 40, 5, 30, 35, 45, 18.07132963, 24.96273236, 85.12912161, 94.99897537, 5.561851831, 7.199504273, 102.5184759, 112.4750941), ('rice', 60, 99, 35, 60, 35, 45, 20.0454142, 26.92995077, 80.12267476, 84.96907151, 5.005306977, 7.868474653, 182.5616319, 298.5601175), ('watermelon', 80, 120, 5, 30, 45, 55, 24.04355803, 26.98603693, 80.02621335, 89.98405233, 6.000975617000001, 6.956508826, 40.12650421, 59.75980023)]
=========================================================
Answer:Here is the report on the range of soil requirements for various crops:

* Apple: Nitrogen (N) ranges from 0 to 40, Phosphorus (P) from 120 to 145, Potassium (K) from 195 to 205, temperature from 21.04 to 24.00 degrees Celsius, humidity from 90.03 to 94.92%, pH from 5.51 to 6.50, and rainfall from 100.12 to 124.98 millimeters.
* Banana: Nitrogen (N) ranges from 80 to 120, Phosphorus (P) from 70 to 95, Potassium (K) from 45 to 55, temperature from 25.01 to 29.91 degrees Celsius, humidity from 75.03 to 84.98%, pH from 5.51 to 6.49, and rainfall from 90.11 to 119.85 millimeters.
* Blackgram: Nitrogen (N) ranges from 20 to 60, Phosphorus (P) from 55 to 80, Potassium (K) from 15 to 25, temperature from 25.10 to 34.95 degrees Celsius, humidity from 60.07 to 69.96%, pH from 6.50 to 7.78, and rainfall from 60.42 to 74.92 millimeters.
* Chickpea: Nitrogen (N) ranges from 20 to 60, Phosphorus (P) from 55 to 80, Potassium (K) from 75 to 85, temperature from 17.02 to 20.99 degrees Celsius, humidity from 14.26 to 20.00%, pH from 5.99 to 8.87, and rainfall from 65.11 to 94.78 millimeters.
* Coconut: Nitrogen (N) ranges from 0 to 40, Phosphorus (P) from 5 to 30, Potassium (K) from 25 to 35, temperature from 25.01 to 29.87 degrees Celsius, humidity from 90.02 to 99.98%, pH from 5.50 to 6.47, and rainfall from 131.10 to 225.63 millimeters.
* Coffee: Nitrogen (N) ranges from 80 to 120, Phosphorus (P) from 15 to 40, Potassium (K) from 25 to 35, temperature from 23.06 to 27.92 degrees Celsius, humidity from 50.05 to 69.95%, pH from 6.02 to 7.49, and rainfall from 115.16 to 199.47 millimeters.
* Cotton: Nitrogen (N) ranges from 100 to 140, Phosphorus (P) from 35 to 60, Potassium (K) from 15 to 25, temperature from 22.00 to 25.99 degrees Celsius, humidity from 75.01 to 84.88%, pH from 5.80 to 8.00, and rainfall from 60.65 to 99.93 millimeters.
* Grapes: Nitrogen (N) ranges from 0 to 40, Phosphorus (P) from 120 to 145, Potassium (K) from 195 to 205, temperature from 8.83 to 41.95 degrees Celsius, humidity from 80.02 to 83.98%, pH from 5.51 to 6.50, and rainfall from 65.01 to 74.91 millimeters.
* Jute: Nitrogen (N) ranges from 60 to 100, Phosphorus (P) from 35 to 60, Potassium (K) from 35 to 45, temperature from 23.10 to 26.99 degrees Celsius, humidity from 70.88 to 89.90%, pH from 6.00 to 7.49, and rainfall from 150.24 to 199.84 millimeters.
* Kidneybeans: Nitrogen (N) ranges from 0 to 40, Phosphorus (P) from 55 to 80, Potassium (K) from 15 to 25, temperature from 15.33 to 24.92 degrees Celsius, humidity from 18.10 to 25.00%, pH from 5.50 to 5.99, and rainfall from 60.28 to 149.74 millimeters.
* Lentil: Nitrogen (N) ranges from 0 to 40, Phosphorus (P) from 55 to 80, Potassium (K) from 15 to 25, temperature from 18.06 to 29.94 degrees Celsius, humidity from 60.09 to 69.92%, pH from 5.92 to 7.84, and rainfall from 35.03 to 54.94 millimeters.
* Maize: Nitrogen (N) ranges from 60 to 100, Phosphorus (P) from 35 to 60, Potassium (K) from 15 to 25, temperature from 18.04 to 26.55 degrees Celsius, humidity from 55.28 to 74.83%, pH from 5.51 to 6.99, and rainfall from 60.65 to 109.75 millimeters.
* Mango: Nitrogen (N) ranges from 0 to 40, Phosphorus (P) from 15 to 40, Potassium (K) from 25 to 35, temperature from 27.00 to 35.99 degrees Celsius, humidity from 45.02 to 54.96%, pH from 4.51 to 6.97, and rainfall from 89.29 to 100.81 millimeters.
* Mothbeans: Nitrogen (N) ranges from 0 to 40, Phosphorus (P) from 35 to 60, Potassium (K) from 15 to 25, temperature from 24.02 to 32.00 degrees Celsius, humidity from 40.01 to 64.96%, pH from 3.50 to 9.94, and rainfall from 30.92 to 74.44 millimeters.
* Mungbean: Nitrogen (N) ranges from 0 to 40, Phosphorus (P) from 35 to 60, Potassium (K) from 15 to 25, temperature from 27.01 to 29.91 degrees Celsius, humidity from 80.03 to 90.00%, pH from 6.22 to 7.20, and rainfall from 36.12 to 59.87 millimeters.
* Muskmelon: Nitrogen (N) ranges from 80 to 120, Phosphorus (P) from 5 to 30, Potassium (K) from 45 to 55, temperature from 27.02 to 29.94 degrees Celsius, humidity from 90.02 to 94.96%, pH from 6.00 to 6.78, and rainfall from 20.21 to 29.87 millimeters.
* Orange: Nitrogen (N) ranges from 0 to 40, Phosphorus (P) from 5 to 15, Potassium (K) from 5 to 15, temperature from 10.01 to 34.91 degrees Celsius, humidity from 90.01 to 94.96%, pH from 6.01 to 8.00, and rainfall from 100.17 to 119.70 millimeters.
* Papaya: Nitrogen (N) ranges from 31 to 70, Phosphorus (P) from 46 to 70, Potassium (K) from 45 to 55, temperature from 23.01 to 43.68 degrees Celsius, humidity from 90.04 to 94.94%, pH from 6.50 to 6.99, and rainfall from 40.35 to 248.86 millimeters.
* Pigeonpeas: Nitrogen (N) ranges from 0 to 40, Phosphorus (P) from 55 to 80, Potassium (K) from 15 to 25, temperature from 18.32 to 36.98 degrees Celsius, humidity from 30.40 to 69.69%, pH from 4.55 to 7.45, and rainfall from 90.05 to 198.83 millimeters.
* Pomegranate: Nitrogen (N) ranges from 0 to 40, Phosphorus (P) from 5 to 30, Potassium (K) from 35 to 45, temperature from 18.07 to 24.96 degrees Celsius, humidity from 85.13 to 94.99%, pH from 5.56 to 7.20, and rainfall from 102.52 to 112.48 millimeters.
* Rice: Nitrogen (N) ranges from 60 to 99, Phosphorus (P) from 35 to 60, Potassium (K) from 35 to 45, temperature from 20.05 to 26.93 degrees Celsius, humidity from 80.12 to 84.97%, pH from 5.01 to 7.87, and rainfall from 182.56 to 298.56 millimeters.
* Watermelon: Nitrogen (N) ranges from 80 to 120, Phosphorus (P) from 5 to 30, Potassium (K) from 45 to 55, temperature from 24.04 to 26.99 degrees Celsius, humidity from 80.03 to 89.98%, pH from 6.00 to 6.96, and rainfall from 40.13 to 59.76 millimeters.
```

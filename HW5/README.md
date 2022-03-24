# CNN-RNN Implementation of Precipitation Nowcasting

Dataset Explaination:

This data contains satellite images in 3 different channels (Cloud Density1, Cloud Density2, Water vapor) covering a 5x5 grid from different parts of Thailand. Additionally each data point contains 5 time steps, so each input data oint has a size of 5x5x5x3 (time, H, W, C) as a features. 

<img src="https://user-images.githubusercontent.com/62899961/159891664-e9f432af-8f5e-4af6-8f4e-e132f4eaea04.png" width="500" height="300">

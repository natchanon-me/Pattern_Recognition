# CNN-RNN Implementation of Precipitation Nowcasting

Download dataset : [Google Drive](https://drive.google.com/file/d/1NWR22fVVE0tO2Q5EbaPPrRKPhUem-jbw/view)

Credit : [ekapolc](https://github.com/ekapolc/pattern_2022/tree/main/HW)

Dataset Explaination:

This data contains satellite images in 3 different channels (Cloud Density1, Cloud Density2, Water vapor) covering a `5x5` grid from different parts of Thailand. Additionally each data point contains 5 time steps, so each input data oint has a size of `5x5x5x3` (time, H, W, C) as a features. 

Since the input is basically image with time-related, the final solution in this work is combining ***CNN*** with ***RNN*** (GRU to be percise) in different scheme shows in below picture<br/><br/>

<p align="center">
  <img src="https://user-images.githubusercontent.com/62899961/159891664-e9f432af-8f5e-4af6-8f4e-e132f4eaea04.png" width="500" height="300">
</p>

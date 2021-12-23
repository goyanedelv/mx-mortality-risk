library(mxmaps)
library(openxlsx)

str(df_mxmunicipio_2020)

write.xlsx(df_mxmunicipio_2020, 'caract_municipio_mx.xlsx')

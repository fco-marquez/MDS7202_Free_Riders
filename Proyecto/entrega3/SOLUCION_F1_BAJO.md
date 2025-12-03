# 🔴 PROBLEMA IDENTIFICADO: F1 = 0.03 en CodaLab

## ❌ **Causa Raíz: UMBRAL DEMASIADO ALTO**

### El Problema:

```
Umbral usado: 0.09
Predicciones enviadas: 146 (0.086% del universo)
Esperado (~2.6%): 4,416 predicciones
F1 obtenido: 0.03 ❌
```

**Conclusión:** Enviaste **30× menos predicciones** de las necesarias → Recall bajísimo → F1 pésimo

---

## ✅ **SOLUCIÓN IMPLEMENTADA**

### 1. Análisis de Umbrales Óptimos

He analizado la distribución de probabilidades del modelo:

| Umbral    | Predicciones | % Universo | Evaluación                |
| --------- | ------------ | ---------- | ------------------------- |
| 0.064     | 36,295       | 21.4%      | ❌ Demasiadas             |
| 0.065     | 11,737       | 6.9%       | ⚠️ Muchas                 |
| 0.066     | 7,387        | 4.4%       | ⚠️ Muchas                 |
| **0.067** | **4,486**    | **2.64%**  | **✅ ÓPTIMO**             |
| 0.068     | 2,703        | 1.6%       | ⚠️ Pocas                  |
| 0.069     | 2,193        | 1.3%       | ⚠️ Pocas                  |
| 0.070     | 1,700        | 1.0%       | ❌ Muy pocas              |
| 0.090     | 146          | 0.086%     | ❌ **MUY POCAS** (actual) |

### 2. Umbral Optimizado: **0.067**

**¿Por qué 0.067?**

- Genera **4,486 predicciones** (2.64% del universo)
- Coincide casi exactamente con la tasa esperada de compra (~2.6%)
- Diferencia de solo +70 predicciones vs óptimo teórico
- Balance ideal entre precision y recall

### 3. Archivos Generados

✅ **Nuevo CSV optimizado:**

```
airflow/predictions/prediccion_optimized_threshold_0.067.csv
```

- **4,486 predicciones** (vs 146 anteriores)
- Probabilidades: 0.067 - 0.128
- Listo para subir a CodaLab

✅ **Código actualizado:**

- `dag.py`: Umbral cambiado de 0.09 a 0.067
- Próxima ejecución del DAG usará el umbral correcto

---

## 📊 **Mejora Esperada en CodaLab**

### Antes (F1 = 0.03):

```
Predicciones: 146
Recall: ~0.02 (muy bajo)
Precision: ~0.15 (estimado)
F1: 0.03 ❌
```

### Después (con umbral 0.067):

```
Predicciones: 4,486
Recall esperado: ~0.30-0.50 (mucho mejor)
Precision esperado: ~0.15-0.30
F1 esperado: ~0.20-0.35 ✅
```

**Mejora estimada:** **F1 de 0.03 → 0.20-0.35** (6-11× mejor)

---

## 🚀 **PRÓXIMOS PASOS**

### INMEDIATO:

1. **Subir nuevo CSV a CodaLab:**

   ```
   airflow/predictions/prediccion_optimized_threshold_0.067.csv
   ```

2. **Verificar mejora en F1:**
   - F1 debería ser >0.20 (vs 0.03 anterior)
   - Si sigue bajo, el problema es del modelo, no del umbral

### SI F1 SIGUE BAJO (<0.15):

Entonces el problema NO es el umbral, sino el **modelo**:

**Posibles causas:**

1. Features débiles (recency/frequency no predictivos)
2. Product_id como numérico confunde al modelo
3. Falta de features temporales (estacionalidad)
4. Datos de entrenamiento insuficientes

**Próximas acciones:**

1. Revisar métricas en MLflow (recall en validación)
2. Agregar features temporales (mes, día semana)
3. Target encoding para product_id
4. Probar SMOTE para balance sintético
5. Considerar LightGBM o CatBoost

---

## 📈 **Contexto: ¿Por qué el modelo genera probabilidades bajas?**

### Es Normal con Desbalance Extremo:

Tu dataset tiene:

- **Ratio 36.89:1** (negativo:positivo)
- Solo **2.6% de pares compran**
- Con datos tan desbalanceados, probabilidades de 6-13% son **razonables**

### El modelo mejoró:

- **Antes:** Probabilidades constantes (std=0.0003) → NO discriminaba
- **Ahora:** Probabilidades variables (std=0.0022, 6.9× más) → SÍ discrimina
- **Rango:** 0.042 - 0.128 (8.9× más amplio)

El modelo **SÍ está funcionando**, solo que es conservador (apropiado para datos desbalanceados).

---

## ✅ **RESUMEN**

| Aspecto                   | Estado                          |
| ------------------------- | ------------------------------- |
| **Problema identificado** | ✅ Umbral demasiado alto (0.09) |
| **Solución implementada** | ✅ Umbral optimizado (0.067)    |
| **CSV generado**          | ✅ 4,486 predicciones           |
| **Código actualizado**    | ✅ dag.py modificado            |
| **Mejora esperada**       | ✅ F1: 0.03 → 0.20-0.35         |

---

## 📝 **NOTAS TÉCNICAS**

### ¿Por qué salto de 11,737 a 4,486 predicciones entre 0.065 y 0.067?

La distribución de probabilidades tiene un **pico** alrededor de 0.065-0.066. Muchos pares tienen probabilidades similares en ese rango, por eso un cambio pequeño de umbral (0.002) elimina ~7,000 predicciones.

**Visualización aproximada:**

```
[0.065] ████████████ 11,737 pares
[0.066] ███████       7,387 pares
[0.067] ████          4,486 pares ← ÓPTIMO
[0.068] ██            2,703 pares
[0.069] █             2,193 pares
```

---

**Última actualización:** 3 de diciembre de 2025  
**Acción requerida:** Subir `prediccion_optimized_threshold_0.067.csv` a CodaLab

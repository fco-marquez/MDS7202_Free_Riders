"""
Gradio Frontend for SODAI Drinks Prediction System
Provides user interface for predictions and recommendations.
"""
import os
import gradio as gr
import requests
import pandas as pd
from typing import Tuple, Optional

# Backend API URL (from environment or default)
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

# ============================================================================
# API Communication Functions
# ============================================================================

def check_backend_health():
    """Check if backend is healthy."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return True, data
        return False, None
    except Exception as e:
        return False, str(e)


def make_prediction(customer_id: int, product_id: int) -> Tuple[str, str]:
    """
    Make a single prediction via backend API.

    Returns
    -------
    tuple
        (result_message, status_message)
    """
    try:
        # Validate inputs
        if not customer_id or not product_id:
            return "❌ Error: Debes ingresar Customer ID y Product ID", "error"

        # Call backend
        response = requests.post(
            f"{BACKEND_URL}/predict",
            json={
                "customer_id": int(customer_id),
                "product_id": int(product_id)
            },
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()

            # Format result
            prediction = "✅ COMPRARÁ" if data['prediction'] == 1 else "❌ NO COMPRARÁ"
            probability = data['probability'] * 100

            result = f"""
## Predicción: {prediction}

**Probabilidad de compra:** {probability:.2f}%

---

### Detalles

- **Customer ID:** {data['customer_id']}
- **Product ID:** {data['product_id']}
- **Semana de predicción:** {data['week']}

### Información del Cliente
- **Tipo:** {data['customer_type']}

### Información del Producto
- **Marca:** {data['product_brand']}
- **Categoría:** {data['product_category']}

---

{'🎯 **Recomendación:** Este cliente tiene alta probabilidad de comprar este producto.' if data['prediction'] == 1 else '💡 **Sugerencia:** Considera otros productos con mayor probabilidad.'}
"""
            return result, "success"

        elif response.status_code == 404:
            return f"❌ Error: Cliente o producto no encontrado en la base de datos", "error"
        else:
            return f"❌ Error: {response.json().get('detail', 'Error desconocido')}", "error"

    except requests.exceptions.Timeout:
        return "❌ Error: Timeout - El servidor tardó demasiado en responder", "error"
    except requests.exceptions.ConnectionError:
        return "❌ Error: No se puede conectar al backend. Verifica que esté ejecutándose.", "error"
    except Exception as e:
        return f"❌ Error inesperado: {str(e)}", "error"


def get_recommendations(customer_id: int, top_n: int = 5) -> Tuple[pd.DataFrame, str]:
    """
    Get top N product recommendations via backend API.

    Returns
    -------
    tuple
        (recommendations_dataframe, status_message)
    """
    try:
        # Validate input
        if not customer_id:
            return pd.DataFrame(), "❌ Error: Debes ingresar un Customer ID"

        # Call backend
        response = requests.post(
            f"{BACKEND_URL}/recommend",
            json={
                "customer_id": int(customer_id),
                "top_n": int(top_n)
            },
            timeout=60
        )

        if response.status_code == 200:
            data = response.json()
            recommendations = data['recommendations']

            # Convert to DataFrame
            df = pd.DataFrame(recommendations)

            # Format columns
            df = df[[
                'rank', 'product_id', 'probability', 'brand',
                'category', 'sub_category', 'segment', 'package', 'size'
            ]]

            # Rename columns in Spanish
            df.columns = [
                'Rank', 'ID Producto', 'Probabilidad', 'Marca',
                'Categoría', 'Subcategoría', 'Segmento', 'Empaque', 'Tamaño'
            ]

            # Format probability as percentage
            df['Probabilidad'] = df['Probabilidad'].apply(lambda x: f"{x*100:.2f}%")

            status = f"✅ Se generaron {len(df)} recomendaciones para el cliente {customer_id}"

            return df, status

        elif response.status_code == 404:
            return pd.DataFrame(), f"❌ Error: Cliente {customer_id} no encontrado en la base de datos"
        else:
            return pd.DataFrame(), f"❌ Error: {response.json().get('detail', 'Error desconocido')}"

    except requests.exceptions.Timeout:
        return pd.DataFrame(), "❌ Error: Timeout - El servidor tardó demasiado en responder"
    except requests.exceptions.ConnectionError:
        return pd.DataFrame(), "❌ Error: No se puede conectar al backend. Verifica que esté ejecutándose."
    except Exception as e:
        return pd.DataFrame(), f"❌ Error inesperado: {str(e)}"


def get_sample_ids() -> Tuple[str, str]:
    """Get sample customer and product IDs from backend."""
    try:
        customers_resp = requests.get(f"{BACKEND_URL}/customers/sample?limit=5", timeout=5)
        products_resp = requests.get(f"{BACKEND_URL}/products/sample?limit=5", timeout=5)

        if customers_resp.status_code == 200 and products_resp.status_code == 200:
            customers = customers_resp.json()['customers']
            products = products_resp.json()['products']

            customer_msg = f"Ejemplos de Customer IDs: {', '.join(map(str, customers))}"
            product_msg = f"Ejemplos de Product IDs: {', '.join(map(str, products))}"

            return customer_msg, product_msg
        else:
            return "No se pudieron obtener ejemplos", "No se pudieron obtener ejemplos"

    except Exception as e:
        return f"Error: {e}", f"Error: {e}"


# ============================================================================
# Gradio Interface
# ============================================================================

def create_interface():
    """Create Gradio interface with tabs."""

    with gr.Blocks(
        title="SODAI Drinks - Predicción de Compras",
        theme=gr.themes.Soft()
    ) as demo:

        gr.Markdown(
            """
            # 🥤 SODAI Drinks - Sistema de Predicción de Compras

            Sistema de Machine Learning para predecir comportamiento de compra de clientes
            y generar recomendaciones de productos.
            """
        )

        # Backend status
        with gr.Row():
            status_btn = gr.Button("🔄 Verificar estado del backend", size="sm")
            status_msg = gr.Textbox(label="Estado", interactive=False, scale=3)

        def check_status():
            healthy, data = check_backend_health()
            if healthy:
                model_info = data.get('model_info', {})
                source = model_info.get('source', 'unknown')
                return f"✅ Backend operativo | Modelo cargado desde: {source}"
            else:
                return f"❌ Backend no disponible: {data}"

        status_btn.click(fn=check_status, outputs=status_msg)

        # Tabs
        with gr.Tabs():

            # ================================================================
            # Tab 1: Single Prediction
            # ================================================================
            with gr.TabItem("🎯 Predicción Individual"):
                gr.Markdown(
                    """
                    ### Predecir si un cliente comprará un producto específico

                    Ingresa el ID del cliente y el ID del producto para obtener la predicción
                    de compra para la próxima semana.
                    """
                )

                with gr.Row():
                    pred_customer_id = gr.Number(
                        label="Customer ID",
                        info="Identificador del cliente",
                        precision=0
                    )
                    pred_product_id = gr.Number(
                        label="Product ID",
                        info="Identificador del producto",
                        precision=0
                    )

                pred_button = gr.Button("🔮 Predecir", variant="primary")

                pred_result = gr.Markdown(label="Resultado")

                # Examples info
                with gr.Accordion("ℹ️ Ver ejemplos de IDs", open=False):
                    examples_btn = gr.Button("Obtener ejemplos")
                    example_customers = gr.Textbox(label="Customer IDs de ejemplo", interactive=False)
                    example_products = gr.Textbox(label="Product IDs de ejemplo", interactive=False)

                    examples_btn.click(
                        fn=get_sample_ids,
                        outputs=[example_customers, example_products]
                    )

                # Examples
                gr.Examples(
                    examples=[
                        [1001, 2001],
                        [1002, 2005],
                        [1003, 2010],
                    ],
                    inputs=[pred_customer_id, pred_product_id],
                    label="Ejemplos rápidos"
                )

                # Wire up prediction
                pred_button.click(
                    fn=lambda c, p: make_prediction(c, p)[0],
                    inputs=[pred_customer_id, pred_product_id],
                    outputs=pred_result
                )

            # ================================================================
            # Tab 2: Recommendations
            # ================================================================
            with gr.TabItem("⭐ Recomendaciones"):
                gr.Markdown(
                    """
                    ### Sistema de Recomendación de Productos

                    Obtén los productos con mayor probabilidad de compra para un cliente específico.
                    El sistema evaluará todos los productos disponibles y te mostrará los Top N.
                    """
                )

                with gr.Row():
                    rec_customer_id = gr.Number(
                        label="Customer ID",
                        info="Identificador del cliente",
                        precision=0
                    )
                    rec_top_n = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="Número de recomendaciones",
                        info="Cantidad de productos a recomendar"
                    )

                rec_button = gr.Button("🎁 Generar Recomendaciones", variant="primary")

                rec_status = gr.Textbox(label="Estado", interactive=False)
                rec_results = gr.Dataframe(
                    label="Productos Recomendados",
                    wrap=True
                )

                # Examples
                gr.Examples(
                    examples=[
                        [1001, 5],
                        [1002, 10],
                        [1003, 3],
                    ],
                    inputs=[rec_customer_id, rec_top_n],
                    label="Ejemplos rápidos"
                )

                # Wire up recommendations
                rec_button.click(
                    fn=get_recommendations,
                    inputs=[rec_customer_id, rec_top_n],
                    outputs=[rec_results, rec_status]
                )

        # Footer
        gr.Markdown(
            """
            ---

            ### 📊 Información del Sistema

            - **Modelo:** XGBoost Classifier optimizado con Optuna
            - **Features:** Clustering geográfico + RFM (Recency, Frequency, Monetary)
            - **Tracking:** MLflow para versionado de modelos
            - **Pipeline:** Airflow para reentrenamiento automático con detección de drift

            *Proyecto desarrollado para MDS7202 - Entrega 2*
            """
        )

    return demo


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Starting SODAI Drinks Prediction Frontend...")
    print(f"Backend URL: {BACKEND_URL}")

    # Create and launch interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

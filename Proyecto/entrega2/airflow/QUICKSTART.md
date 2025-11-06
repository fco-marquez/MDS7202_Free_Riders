# 🚀 Quick Start - SodAI Pipeline

## Inicio Rápido con Docker (5 minutos)

### Prerrequisitos
- Docker Desktop instalado: https://www.docker.com/products/docker-desktop/

### Pasos

```bash
# 1. Navegar al directorio
cd C:\Users\fmarq\DCC\MDS\MDS7202_Laboratorio\MDS7202_Free_Riders\Proyecto\entrega2\airflow

# 2. Iniciar servicios (primera vez tarda 5-10 min)
docker-compose up -d

# 3. Esperar que servicios estén listos (~60 segundos)
docker-compose ps

# 4. Acceder a Airflow UI
# http://localhost:8080
# Usuario: admin
# Password: admin

# 5. Activar y ejecutar el DAG 'sodai_prediction_pipeline'
```

### Interfaces Disponibles

- **Airflow UI**: http://localhost:8080
- **MLflow UI**: http://localhost:5000

### Comandos Útiles

```bash
# Ver logs en tiempo real
docker-compose logs -f

# Detener servicios
docker-compose stop

# Iniciar servicios detenidos
docker-compose start

# Limpiar todo
docker-compose down -v
```

---

## Generar Datos de Prueba (para video/demo)

```bash
# Opción 1: Desde contenedor Docker
docker-compose exec airflow python generate_test_data.py --mode add_weeks --weeks 2 --noise 0.3

# Opción 2: Desde host (si tienes Python)
python generate_test_data.py --mode add_weeks --weeks 2 --noise 0.3

# Ver resumen de datos
python generate_test_data.py --mode summary
```

---

## Estructura de Archivos

```
airflow/
├── Dockerfile                 # Imagen Docker
├── docker-compose.yml         # Orquestación
├── requirements.txt           # Dependencias Python
├── dags/                      # DAGs y módulos
│   ├── dag.py                 # DAG principal
│   ├── drift_detector.py      # Drift detection
│   ├── train_module.py        # Entrenamiento
│   └── ...
├── data/                      # Datos (raw y processed)
├── README.md                  # Documentación completa
├── VIDEO_GUIDE.md             # Guía para grabar video
└── QUICKSTART.md              # Este archivo
```

---

## Troubleshooting

**Problema: Puerto 8080 ocupado**
```bash
# Opción 1: Cerrar servicio que usa el puerto
# Opción 2: Cambiar puerto en docker-compose.yml
ports:
  - "8081:8080"
```

**Problema: Servicios no inician**
```bash
docker-compose logs
# Verificar RAM asignada a Docker (mínimo 4GB)
```

**Problema: DAG no aparece**
```bash
# Esperar 30 segundos y refrescar
# Verificar logs
docker-compose logs airflow
```

---

## Documentación Completa

Ver `README.md` para documentación detallada.

Ver `VIDEO_GUIDE.md` para guía de grabación del video.

---

**Equipo Free Riders - MDS7202**

# LinearRegression - Modern C++20 Linear Regression Library

[🇩🇪 Deutsch](#deutsch) | [🇪🇸 Español](#español) | [🇬🇧 English](#english)

---

## Deutsch

### Überblick
Eine moderne, template-basierte Implementierung der einfachen linearen Regression mit C++20-Features und statistischen Analysefunktionen. Diese Bibliothek bietet professionelle Regressionsanalyse mit Boost-Integration für präzise Konfidenzintervall-Berechnungen.

### Features
- **Modernes C++20**: Nutzt Concepts, `std::span`, `std::transform_reduce` und `[[nodiscard]]`
- **Template-Basiert**: Generische Implementierung für Gleitkommatypen (float, double, long double)
- **Boost-Integration**: Verwendet Boost.Math für präzise Student-t-Verteilungs-Quantile
- **Parallele Ausführung**: Nutzt `std::execution::par` für optimierte Performance bei großen Datensätzen
- **Umfassende Statistik**: Berechnet Regressionskoeffizienten, Korrelation, SSE und Konfidenzintervalle
- **Speichereffizient**: Arbeitet mit `std::span` zur Vermeidung unnötiger Kopien
- **Typsicher**: Umfangreicher Einsatz von `std::floating_point` Concept für Compile-Time-Sicherheit
- **Header-Only-Kompatibel**: Geeignet für Header-Only-Library-Integration

### Projektstruktur

```
LinearRegression/
├── LinearRegression/
│   ├── linreg.h              # Haupt-Regressionsimplementierung
│   ├── stats.h               # Statistische Hilfsfunktionen (mit parallel execution)
│   ├── span_compatible.h     # Helper-Utilities und Concepts
│   └── LinearRegression.cpp  # Beispielverwendung mit gnuplot-Visualisierung
├── Documentation/            # Zusätzliche Dokumentation
├── LinearRegression.slnx    # Visual Studio Solution
└── README.md                # Diese Datei
```

### Mathematischer Hintergrund

Die Implementierung passt ein lineares Modell der Form an:

```
y = β₀ + β₁x + ε
```

**Wobei:**
- **β₀** (beta0): y-Achsenabschnitt
- **β₁** (beta1): Steigungskoeffizient
- **ρ** (rho): Pearson-Korrelationskoeffizient
- **ε**: Fehlerterm

**Berechnungen:**
1. **Quadratsummen**: Sxx = Σ(xᵢ - x̄)², Syy = Σ(yᵢ - ȳ)², Sxy = Σ(xᵢ - x̄)(yᵢ - ȳ)
2. **Steigung**: β₁ = Sxy / Sxx
3. **Achsenabschnitt**: β₀ = ȳ - β₁x̄
4. **Korrelation**: ρ = Sxy / √(Sxx × Syy)
5. **SSE**: Summe der quadrierten Fehler
6. **Konfidenzintervall**: Verwendet Student-t-Verteilung mit n-2 Freiheitsgraden
7. **Bestimmtheitsmaß**: R² = ρ²

### Installation

#### Voraussetzungen
- C++20-kompatibler Compiler (GCC 10+, Clang 10+, MSVC 2019+) mit TBB-Unterstützung
- Boost-Bibliotheken (speziell Boost.Math)
- Intel TBB (Threading Building Blocks) für parallele Ausführung

#### Abhängigkeiten

```bash
# Ubuntu/Debian
sudo apt-get install libboost-math-dev libtbb-dev

# macOS mit Homebrew
brew install boost tbb

# Windows (vcpkg)
vcpkg install boost-math tbb
```

#### Klonen und Kompilieren

```bash
git clone https://github.com/Haasrobertgmxnet/LinearRegression.git
cd LinearRegression

# Mit g++ (mit TBB für parallele Ausführung)
g++ -std=c++20 -I/pfad/zu/boost -o regression_example LinearRegression.cpp -ltbb

# Für Windows mit MSVC
# Stelle sicher, dass die Projekt-Einstellungen TBB einschließen
```

### Grundlegende Verwendung

```cpp
#include "linreg.h"
#include <vector>
#include <iostream>

int main() {
    // Daten vorbereiten
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {2.1, 3.9, 6.2, 7.8, 10.1};
    
    // Modell anpassen
    auto result = LinearRegression::fit(x, y);
    
    // Regressionsparameter abrufen
    std::cout << "Achsenabschnitt (β₀): " << result.beta0 << std::endl;
    std::cout << "Steigung (β₁): " << result.beta1 << std::endl;
    std::cout << "Korrelation (ρ): " << result.rho << std::endl;
    
    // Bestimmtheitsmaß berechnen
    double r_squared = LinearRegression::coeff_of_determination(result);
    std::cout << "R² = " << r_squared << std::endl;
    
    // 95% Konfidenzintervall für die Steigung berechnen
    auto [lower, upper] = LinearRegression::ci_slope(result, 0.05);
    std::cout << "95% KI: [" << lower << ", " << upper << "]" << std::endl;
    
    return 0;
}
```

### API-Referenz

#### Struct: `FitResult<T>`

Container für Regressionsergebnisse.

**Member:**
- `T beta0` - y-Achsenabschnitt
- `T beta1` - Steigungskoeffizient
- `T rho` - Pearson-Korrelationskoeffizient
- `T sxx`, `syy`, `sxy` - Quadratsummen und Kreuzprodukte
- `T sse` - Summe der quadrierten Fehler
- `std::size_t n` - Anzahl der Datenpunkte

#### Funktion: `fit`

```cpp
template <typename T>
    requires std::is_floating_point_v<T>
[[nodiscard]]
FitResult<T> fit(std::span<const T> x, std::span<const T> y)
```
bzw. Container-Überladung
```cpp
template <Helper::SpanCompatible C>
[[nodiscard]]
FitResult<double> fit(const C& x, const C& y)
```

Passt ein lineares Regressionsmodell an die Daten an.

**Anforderungen:**
- x und y müssen gleiche Größe haben
- Mindestens 3 Datenpunkte erforderlich
- x-Werte dürfen nicht alle identisch sein

**Hinweis:** Verwendet parallele Ausführung (`std::execution::par`) für Mean-Centering und Dot-Product-Berechnungen.

#### Funktion: `ci_slope`

```cpp
template <typename T>
    requires std::is_floating_point_v<T>
[[nodiscard]]
std::pair<T, T> ci_slope(const FitResult<T>& fitResult, const T alpha)
```
bzw. double-Spezialisierung
```cpp
[[nodiscard]]
inline std::pair<double, double>
ci_slope(const FitResult<double>& fitResult, const double alpha)
```

Berechnet das Konfidenzintervall für den Steigungskoeffizienten.

**Parameter:**
- `fitResult` - Ergebnis der `fit()`-Funktion
- `alpha` - Signifikanzniveau (z.B. 0.05 für 95% KI, 0.01 für 99% KI)

#### Funktion: `coeff_of_determination`

```cpp
template <typename T>
    requires std::is_floating_point_v<T>
[[nodiscard]]
T coeff_of_determination(const FitResult<T>& fitResult)
```
bzw. double-Spezialisierung
```cpp
[[nodiscard]]
inline double coeff_of_determination(const FitResult<double>& fitResult)
```

Berechnet das Bestimmtheitsmaß (R²).

**Rückgabe:** R²-Wert (0 bis 1) zur Messung der Anpassungsgüte

**Interpretation:**
- R² = 1.0: Perfekte Anpassung
- R² > 0.9: Ausgezeichnete Anpassung
- R² > 0.7: Gute Anpassung
- R² < 0.4: Schwache Anpassung

**Beispiel:**
```cpp
auto result = LinearRegression::fit(x, y);
double r_squared = LinearRegression::coeff_of_determination(result);
std::cout << "Modell erklärt " << (r_squared * 100) << "% der Varianz\n";
```

### Stats-Namespace Funktionen

#### `Stats::mean`
```cpp
template <std::floating_point T>
[[nodiscard]]
T mean(std::span<const T> data)
```
Berechnet arithmetisches Mittel mit paralleler Ausführung (`std::reduce` mit `std::execution::par`).

#### `Stats::shift`
```cpp
template <std::floating_point T>
[[nodiscard]]
std::vector<T> shift(std::span<const T> x)
```
Zentriert Daten um Mittelwert (mean-centering) mit paralleler Transformation.

#### `Stats::inner_product`
```cpp
template <std::floating_point T>
[[nodiscard]]
T inner_product(std::span<const T> x, std::span<const T> y)
```
Berechnet Dot-Product mit paralleler Ausführung (`std::transform_reduce` mit `std::execution::par`).

### Fortgeschrittene Verwendung

#### Arbeiten mit verschiedenen Datentypen

```cpp
// Mit float
std::vector<float> x_f = {1.0f, 2.0f, 3.0f};
std::vector<float> y_f = {2.0f, 4.0f, 6.0f};
auto result_f = LinearRegression::fit(x_f, y_f);

// Mit double (empfohlen)
std::vector<double> x_d = {1.0, 2.0, 3.0};
std::vector<double> y_d = {2.0, 4.0, 6.0};
auto result_d = LinearRegression::fit(x_d, y_d);

// Mit long double (maximale Präzision)
std::vector<long double> x_ld = {1.0L, 2.0L, 3.0L};
std::vector<long double> y_ld = {2.0L, 4.0L, 6.0L};
auto result_ld = LinearRegression::fit(x_ld, y_ld);
```

#### std::span direkt verwenden

```cpp
double x_data[] = {1.0, 2.0, 3.0, 4.0};
double y_data[] = {2.0, 4.0, 6.0, 8.0};

std::span<const double> x_span{x_data};
std::span<const double> y_span{y_data};

auto result = LinearRegression::fit(x_span, y_span);
```

#### Visualisierung mit gnuplot

Das mitgelieferte `LinearRegression.cpp` zeigt, wie man Ergebnisse mit gnuplot visualisiert:

```cpp
#include "linreg.h"
#include <vector>

int main() {
    std::vector<double> x = {1,2,3,4,5,6,7,8,9,10};
    std::vector<double> y = {3.1,5.0,7.2,9.1,10.0,13.2,15.5,16.5,19.0,21.3};
    
    auto result = LinearRegression::fit(x, y);
    auto [lower, upper] = LinearRegression::ci_slope(result, 0.05);
    
    // plot_chart() Funktion erstellt gnuplot-Visualisierung mit:
    // - Datenpunkten
    // - Regressionsgerade
    // - Konfidenzintervall-Band
    plot_chart(x, y);
    
    return 0;
}
```

### Performance-Hinweise

**Parallele Ausführung:**
- Die Stats-Funktionen nutzen `std::execution::par` für große Datensätze
- Bei kleinen Datensätzen (<1000 Punkte) kann der Overhead die Vorteile überwiegen
- TBB (Threading Building Blocks) muss zur Link-Zeit verfügbar sein

**Rundungsfehler:**
- Parallele Ausführung kann zu minimalen Unterschieden in Gleitkomma-Rundungen führen
- Für exakt reproduzierbare Ergebnisse verwende sequentielle Ausführung

### Lizenz

Dieses Projekt ist Open Source und unter der MIT-Lizenz verfügbar.

### Autor

**Haasrobertgmxnet**
- GitHub: [@Haasrobertgmxnet](https://github.com/Haasrobertgmxnet)

### Mitwirken

Issues und Pull Requests sind willkommen! Bitte öffne ein Issue für größere Änderungen.

---

## Español

### Descripción General
Una implementación moderna basada en templates de regresión lineal simple utilizando características de C++20 con capacidades de análisis estadístico. Esta biblioteca proporciona análisis de regresión de nivel profesional con integración de Boost para cálculos precisos de intervalos de confianza.

### Características
- **C++20 Moderno**: Utiliza concepts, `std::span`, `std::transform_reduce` y `[[nodiscard]]`
- **Basado en Templates**: Implementación genérica para tipos de punto flotante (float, double, long double)
- **Integración Boost**: Usa Boost.Math para cuantiles precisos de la distribución t de Student
- **Ejecución Paralela**: Utiliza `std::execution::par` para rendimiento optimizado en conjuntos grandes de datos
- **Estadísticas Completas**: Calcula coeficientes de regresión, correlación, SSE e intervalos de confianza
- **Eficiencia de Memoria**: Trabaja con `std::span` para evitar copias innecesarias
- **Seguridad de Tipos**: Uso extensivo del concept `std::floating_point` para seguridad en tiempo de compilación
- **Compatible Header-Only**: Adecuado para integración de biblioteca header-only

### Estructura del Proyecto

```
LinearRegression/
├── LinearRegression/
│   ├── linreg.h              # Implementación principal de regresión
│   ├── stats.h               # Funciones de utilidad estadística (con ejecución paralela)
│   ├── span_compatible.h     # Utilidades helper y concepts
│   └── LinearRegression.cpp  # Ejemplo de uso con visualización gnuplot
├── Documentation/            # Documentación adicional
├── LinearRegression.slnx    # Solución de Visual Studio
└── README.md                # Este archivo
```

### Fundamento Matemático

La implementación ajusta un modelo lineal de la forma:

```
y = β₀ + β₁x + ε
```

**Donde:**
- **β₀** (beta0): intersección con el eje y
- **β₁** (beta1): coeficiente de pendiente
- **ρ** (rho): coeficiente de correlación de Pearson
- **ε**: término de error

**Cálculos:**
1. **Sumas de Cuadrados**: Sxx = Σ(xᵢ - x̄)², Syy = Σ(yᵢ - ȳ)², Sxy = Σ(xᵢ - x̄)(yᵢ - ȳ)
2. **Pendiente**: β₁ = Sxy / Sxx
3. **Intersección**: β₀ = ȳ - β₁x̄
4. **Correlación**: ρ = Sxy / √(Sxx × Syy)
5. **SSE**: Suma de errores cuadrados
6. **Intervalo de Confianza**: Usa distribución t de Student con n-2 grados de libertad
7. **Coeficiente de determinación**: R² = ρ²

### Instalación

#### Requisitos Previos
- Compilador compatible con C++20 (GCC 10+, Clang 10+, MSVC 2019+) con soporte TBB
- Bibliotecas Boost (específicamente Boost.Math)
- Intel TBB (Threading Building Blocks) para ejecución paralela

#### Dependencias

```bash
# Ubuntu/Debian
sudo apt-get install libboost-math-dev libtbb-dev

# macOS con Homebrew
brew install boost tbb

# Windows (vcpkg)
vcpkg install boost-math tbb
```

#### Clonar y Compilar

```bash
git clone https://github.com/Haasrobertgmxnet/LinearRegression.git
cd LinearRegression

# Usando g++ (con TBB para ejecución paralela)
g++ -std=c++20 -I/ruta/a/boost -o regression_example LinearRegression.cpp -ltbb

# Para Windows con MSVC
# Asegúrate de que la configuración del proyecto incluya TBB
```

### Uso Básico

```cpp
#include "linreg.h"
#include <vector>
#include <iostream>

int main() {
    // Preparar datos
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {2.1, 3.9, 6.2, 7.8, 10.1};
    
    // Ajustar el modelo
    auto result = LinearRegression::fit(x, y);
    
    // Acceder a parámetros de regresión
    std::cout << "Intersección (β₀): " << result.beta0 << std::endl;
    std::cout << "Pendiente (β₁): " << result.beta1 << std::endl;
    std::cout << "Correlación (ρ): " << result.rho << std::endl;
    
    // Calcular coeficiente de determinación
    double r_squared = LinearRegression::coeff_of_determination(result);
    std::cout << "R² = " << r_squared << std::endl;
    
    // Calcular intervalo de confianza del 95% para la pendiente
    auto [lower, upper] = LinearRegression::ci_slope(result, 0.05);
    std::cout << "IC 95%: [" << lower << ", " << upper << "]" << std::endl;
    
    return 0;
}
```

### Referencia de la API

#### Struct: `FitResult<T>`

Contenedor para resultados de regresión.

**Miembros:**
- `T beta0` - intersección con el eje y
- `T beta1` - coeficiente de pendiente
- `T rho` - coeficiente de correlación de Pearson
- `T sxx`, `syy`, `sxy` - sumas de cuadrados y productos cruzados
- `T sse` - suma de errores cuadrados
- `std::size_t n` - número de puntos de datos

#### Función: `fit`

```cpp
template <typename T>
    requires std::is_floating_point_v<T>
[[nodiscard]]
FitResult<T> fit(std::span<const T> x, std::span<const T> y)
```
o sobrecarga de contenedor
```cpp
template <Helper::SpanCompatible C>
[[nodiscard]]
FitResult<double> fit(const C& x, const C& y)
```

Ajusta un modelo de regresión lineal a los datos.

**Requisitos:**
- x e y deben tener el mismo tamaño
- Se requieren mínimo 3 puntos de datos
- Los valores de x no pueden ser todos idénticos

**Nota:** Utiliza ejecución paralela (`std::execution::par`) para mean-centering y cálculos de producto punto.

#### Función: `ci_slope`

```cpp
template <typename T>
    requires std::is_floating_point_v<T>
[[nodiscard]]
std::pair<T, T> ci_slope(const FitResult<T>& fitResult, const T alpha)
```
o especialización double
```cpp
[[nodiscard]]
inline std::pair<double, double>
ci_slope(const FitResult<double>& fitResult, const double alpha)
```

Calcula el intervalo de confianza para el coeficiente de pendiente.

**Parámetros:**
- `fitResult` - Resultado de la función `fit()`
- `alpha` - Nivel de significancia (ej., 0.05 para IC 95%, 0.01 para IC 99%)

#### Función: `coeff_of_determination`

```cpp
template <typename T>
    requires std::is_floating_point_v<T>
[[nodiscard]]
T coeff_of_determination(const FitResult<T>& fitResult)
```
o especialización double
```cpp
[[nodiscard]]
inline double coeff_of_determination(const FitResult<double>& fitResult)
```

Calcula el coeficiente de determinación (R²).

**Retorna:** Valor R² (0 a 1) que mide la bondad de ajuste

**Interpretación:**
- R² = 1.0: Ajuste perfecto
- R² > 0.9: Ajuste excelente
- R² > 0.7: Buen ajuste
- R² < 0.4: Ajuste débil

**Ejemplo:**
```cpp
auto result = LinearRegression::fit(x, y);
double r_squared = LinearRegression::coeff_of_determination(result);
std::cout << "El modelo explica " << (r_squared * 100) << "% de la varianza\n";
```

### Funciones del Namespace Stats

#### `Stats::mean`
```cpp
template <std::floating_point T>
[[nodiscard]]
T mean(std::span<const T> data)
```
Calcula la media aritmética con ejecución paralela (`std::reduce` con `std::execution::par`).

#### `Stats::shift`
```cpp
template <std::floating_point T>
[[nodiscard]]
std::vector<T> shift(std::span<const T> x)
```
Centra datos alrededor de la media (mean-centering) con transformación paralela.

#### `Stats::inner_product`
```cpp
template <std::floating_point T>
[[nodiscard]]
T inner_product(std::span<const T> x, std::span<const T> y)
```
Calcula producto punto con ejecución paralela (`std::transform_reduce` con `std::execution::par`).

### Uso Avanzado

#### Trabajar con Diferentes Tipos de Datos

```cpp
// Usando float
std::vector<float> x_f = {1.0f, 2.0f, 3.0f};
std::vector<float> y_f = {2.0f, 4.0f, 6.0f};
auto result_f = LinearRegression::fit(x_f, y_f);

// Usando double (recomendado)
std::vector<double> x_d = {1.0, 2.0, 3.0};
std::vector<double> y_d = {2.0, 4.0, 6.0};
auto result_d = LinearRegression::fit(x_d, y_d);

// Usando long double (máxima precisión)
std::vector<long double> x_ld = {1.0L, 2.0L, 3.0L};
std::vector<long double> y_ld = {2.0L, 4.0L, 6.0L};
auto result_ld = LinearRegression::fit(x_ld, y_ld);
```

#### Usar std::span Directamente

```cpp
double x_data[] = {1.0, 2.0, 3.0, 4.0};
double y_data[] = {2.0, 4.0, 6.0, 8.0};

std::span<const double> x_span{x_data};
std::span<const double> y_span{y_data};

auto result = LinearRegression::fit(x_span, y_span);
```

#### Visualización con gnuplot

El archivo `LinearRegression.cpp` incluido muestra cómo visualizar resultados con gnuplot:

```cpp
#include "linreg.h"
#include <vector>

int main() {
    std::vector<double> x = {1,2,3,4,5,6,7,8,9,10};
    std::vector<double> y = {3.1,5.0,7.2,9.1,10.0,13.2,15.5,16.5,19.0,21.3};
    
    auto result = LinearRegression::fit(x, y);
    auto [lower, upper] = LinearRegression::ci_slope(result, 0.05);
    
    // La función plot_chart() crea visualización gnuplot con:
    // - Puntos de datos
    // - Línea de regresión
    // - Banda de intervalo de confianza
    plot_chart(x, y);
    
    return 0;
}
```

### Notas de Rendimiento

**Ejecución Paralela:**
- Las funciones Stats utilizan `std::execution::par` para conjuntos grandes de datos
- En conjuntos pequeños (<1000 puntos) el overhead puede superar los beneficios
- TBB (Threading Building Blocks) debe estar disponible en tiempo de enlace

**Errores de Redondeo:**
- La ejecución paralela puede conducir a diferencias mínimas en redondeo de punto flotante
- Para resultados exactamente reproducibles, use ejecución secuencial

### Licencia

Este proyecto es de código abierto y está disponible bajo la Licencia MIT.

### Autor

**Haasrobertgmxnet**
- GitHub: [@Haasrobertgmxnet](https://github.com/Haasrobertgmxnet)

### Contribuciones

¡Issues y Pull Requests son bienvenidos! Por favor, abre un issue para cambios mayores.

---

## English

### Overview
A modern, template-based implementation of simple linear regression using C++20 features with statistical analysis capabilities. This library provides professional-grade regression analysis with Boost integration for accurate confidence interval calculations.

### Features
- **Modern C++20**: Utilizes concepts, `std::span`, `std::transform_reduce`, and `[[nodiscard]]`
- **Template-Based**: Generic implementation for floating-point types (float, double, long double)
- **Boost Integration**: Uses Boost.Math for accurate Student's t-distribution quantiles
- **Parallel Execution**: Leverages `std::execution::par` for optimized performance on large datasets
- **Comprehensive Statistics**: Calculates regression coefficients, correlation, SSE, and confidence intervals
- **Memory Efficient**: Works with `std::span` to avoid unnecessary copies
- **Type Safe**: Extensive use of `std::floating_point` concept for compile-time safety
- **Header-Only Compatible**: Suitable for header-only library integration

### Project Structure

```
LinearRegression/
├── LinearRegression/
│   ├── linreg.h              # Main regression implementation
│   ├── stats.h               # Statistical utility functions (with parallel execution)
│   ├── span_compatible.h     # Helper utilities and concepts
│   └── LinearRegression.cpp  # Example usage with gnuplot visualization
├── Documentation/            # Additional documentation
├── LinearRegression.slnx    # Visual Studio solution
└── README.md                # This file
```

### Mathematical Foundation

The implementation fits a linear model of the form:

```
y = β₀ + β₁x + ε
```

**Where:**
- **β₀** (beta0): y-intercept
- **β₁** (beta1): slope coefficient
- **ρ** (rho): Pearson correlation coefficient
- **ε**: error term

**Calculations:**
1. **Sums of Squares**: Sxx = Σ(xᵢ - x̄)², Syy = Σ(yᵢ - ȳ)², Sxy = Σ(xᵢ - x̄)(yᵢ - ȳ)
2. **Slope**: β₁ = Sxy / Sxx
3. **Intercept**: β₀ = ȳ - β₁x̄
4. **Correlation**: ρ = Sxy / √(Sxx × Syy)
5. **SSE**: Sum of squared errors
6. **Confidence Interval**: Uses Student's t-distribution with n-2 degrees of freedom
7. **Coefficient of determination**: R² = ρ²

### Installation

#### Prerequisites
- C++20 compatible compiler (GCC 10+, Clang 10+, MSVC 2019+) with TBB support
- Boost libraries (specifically Boost.Math)
- Intel TBB (Threading Building Blocks) for parallel execution

#### Dependencies

```bash
# Ubuntu/Debian
sudo apt-get install libboost-math-dev libtbb-dev

# macOS with Homebrew
brew install boost tbb

# Windows (vcpkg)
vcpkg install boost-math tbb
```

#### Clone and Build

```bash
git clone https://github.com/Haasrobertgmxnet/LinearRegression.git
cd LinearRegression

# Using g++ (with TBB for parallel execution)
g++ -std=c++20 -I/path/to/boost -o regression_example LinearRegression.cpp -ltbb

# For Windows with MSVC
# Ensure project settings include TBB
```

### Basic Usage

```cpp
#include "linreg.h"
#include <vector>
#include <iostream>

int main() {
    // Prepare data
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {2.1, 3.9, 6.2, 7.8, 10.1};
    
    // Fit the model
    auto result = LinearRegression::fit(x, y);
    
    // Access regression parameters
    std::cout << "Intercept (β₀): " << result.beta0 << std::endl;
    std::cout << "Slope (β₁): " << result.beta1 << std::endl;
    std::cout << "Correlation (ρ): " << result.rho << std::endl;
    
    // Calculate coefficient of determination
    double r_squared = LinearRegression::coeff_of_determination(result);
    std::cout << "R² = " << r_squared << std::endl;
    
    // Calculate 95% confidence interval for the slope
    auto [lower, upper] = LinearRegression::ci_slope(result, 0.05);
    std::cout << "95% CI: [" << lower << ", " << upper << "]" << std::endl;
    
    return 0;
}
```

### API Reference

#### Struct: `FitResult<T>`

Container for regression results.

**Members:**
- `T beta0` - y-intercept
- `T beta1` - slope coefficient
- `T rho` - Pearson correlation coefficient
- `T sxx`, `syy`, `sxy` - sums of squares and cross-products
- `T sse` - sum of squared errors
- `std::size_t n` - number of data points

#### Function: `fit`

```cpp
template <typename T>
    requires std::is_floating_point_v<T>
[[nodiscard]]
FitResult<T> fit(std::span<const T> x, std::span<const T> y)
```
or container overload
```cpp
template <Helper::SpanCompatible C>
[[nodiscard]]
FitResult<double> fit(const C& x, const C& y)
```

Fits a linear regression model to the data.

**Requirements:**
- x and y must have the same size
- Minimum 3 data points required
- x values must not all be identical

**Note:** Uses parallel execution (`std::execution::par`) for mean-centering and dot-product calculations.

#### Function: `ci_slope`

```cpp
template <typename T>
    requires std::is_floating_point_v<T>
[[nodiscard]]
std::pair<T, T> ci_slope(const FitResult<T>& fitResult, const T alpha)
```
or double specialization
```cpp
[[nodiscard]]
inline std::pair<double, double>
ci_slope(const FitResult<double>& fitResult, const double alpha)
```

Calculates confidence interval for the slope coefficient.

**Parameters:**
- `fitResult` - Result from `fit()` function
- `alpha` - Significance level (e.g., 0.05 for 95% CI, 0.01 for 99% CI)

#### Function: `coeff_of_determination`

```cpp
template <typename T>
    requires std::is_floating_point_v<T>
[[nodiscard]]
T coeff_of_determination(const FitResult<T>& fitResult)
```
or double specialization
```cpp
[[nodiscard]]
inline double coeff_of_determination(const FitResult<double>& fitResult)
```

Calculates the coefficient of determination (R²).

**Returns:** R² value (0 to 1) measuring goodness of fit

**Interpretation:**
- R² = 1.0: Perfect fit
- R² > 0.9: Excellent fit
- R² > 0.7: Good fit
- R² < 0.4: Weak fit

**Example:**
```cpp
auto result = LinearRegression::fit(x, y);
double r_squared = LinearRegression::coeff_of_determination(result);
std::cout << "Model explains " << (r_squared * 100) << "% of variance\n";
```

### Stats Namespace Functions

#### `Stats::mean`
```cpp
template <std::floating_point T>
[[nodiscard]]
T mean(std::span<const T> data)
```
Computes arithmetic mean with parallel execution (`std::reduce` with `std::execution::par`).

#### `Stats::shift`
```cpp
template <std::floating_point T>
[[nodiscard]]
std::vector<T> shift(std::span<const T> x)
```
Centers data around mean (mean-centering) with parallel transformation.

#### `Stats::inner_product`
```cpp
template <std::floating_point T>
[[nodiscard]]
T inner_product(std::span<const T> x, std::span<const T> y)
```
Computes dot product with parallel execution (`std::transform_reduce` with `std::execution::par`).

### Advanced Usage

#### Working with Different Data Types

```cpp
// Using float
std::vector<float> x_f = {1.0f, 2.0f, 3.0f};
std::vector<float> y_f = {2.0f, 4.0f, 6.0f};
auto result_f = LinearRegression::fit(x_f, y_f);

// Using double (recommended)
std::vector<double> x_d = {1.0, 2.0, 3.0};
std::vector<double> y_d = {2.0, 4.0, 6.0};
auto result_d = LinearRegression::fit(x_d, y_d);

// Using long double (maximum precision)
std::vector<long double> x_ld = {1.0L, 2.0L, 3.0L};
std::vector<long double> y_ld = {2.0L, 4.0L, 6.0L};
auto result_ld = LinearRegression::fit(x_ld, y_ld);
```

#### Using std::span Directly

```cpp
double x_data[] = {1.0, 2.0, 3.0, 4.0};
double y_data[] = {2.0, 4.0, 6.0, 8.0};

std::span<const double> x_span{x_data};
std::span<const double> y_span{y_data};

auto result = LinearRegression::fit(x_span, y_span);
```

#### Visualization with gnuplot

The included `LinearRegression.cpp` demonstrates how to visualize results with gnuplot:

```cpp
#include "linreg.h"
#include <vector>

int main() {
    std::vector<double> x = {1,2,3,4,5,6,7,8,9,10};
    std::vector<double> y = {3.1,5.0,7.2,9.1,10.0,13.2,15.5,16.5,19.0,21.3};
    
    auto result = LinearRegression::fit(x, y);
    auto [lower, upper] = LinearRegression::ci_slope(result, 0.05);
    
    // plot_chart() function creates gnuplot visualization with:
    // - Data points
    // - Regression line
    // - Confidence interval band
    plot_chart(x, y);
    
    return 0;
}
```

### Performance Notes

**Parallel Execution:**
- Stats functions use `std::execution::par` for large datasets
- For small datasets (<1000 points), overhead may outweigh benefits
- TBB (Threading Building Blocks) must be available at link time

**Rounding Errors:**
- Parallel execution may lead to minimal floating-point rounding differences
- For exactly reproducible results, use sequential execution

### License

This project is open source and available under the MIT License.

### Author

**Haasrobertgmxnet**
- GitHub: [@Haasrobertgmxnet](https://github.com/Haasrobertgmxnet)

### Contributions

Issues and Pull Requests are welcome! Please open an issue for major changes.

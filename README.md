# LinearRegression - Modern C++20 Linear Regression Library

[🇩🇪 Deutsch](#deutsch) | [🇪🇸 Español](#español) | [🇬🇧 English](#english)

---

## Deutsch

### Überblick
Eine moderne, template-basierte Implementierung der einfachen linearen Regression mit C++20-Features und statistischen Analysefunktionen. Diese Bibliothek bietet professionelle Regressionsanalyse mit Boost-Integration für präzise Konfidenzintervall-Berechnungen.

### Features
- **Modernes C++20**: Nutzt Concepts, `std::span`, `std::transform_reduce` und `[[nodiscard]]`
- **Template-Basiert**: Generische Implementierung für verschiedene numerische Typen (float, double, long double)
- **Boost-Integration**: Verwendet Boost.Math für präzise Student-t-Verteilungs-Quantile
- **Umfassende Statistik**: Berechnet Regressionskoeffizienten, Korrelation, SSE und Konfidenzintervalle
- **Speichereffizient**: Arbeitet mit `std::span` zur Vermeidung unnötiger Kopien
- **Typsicher**: Umfangreicher Einsatz von Concepts und `static_assert` für Compile-Time-Sicherheit
- **Header-Only-Kompatibel**: Geeignet für Header-Only-Library-Integration

### Projektstruktur

```
LinearRegression/
├── LinearRegression/
│   ├── linreg.h      # Haupt-Regressionsimplementierung
│   ├── stats.h                 # Statistische Hilfsfunktionen
│   ├── span_compatible.h       # Helper-Utilities und Concepts
│   └── LinearRegression.cpp               # Beispielverwendung (falls vorhanden)
├── Documentation/              # Zusätzliche Dokumentation
├── LinearRegression.slnx      # Visual Studio Solution
└── README.md                  # Diese Datei
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
- C++20-kompatibler Compiler (GCC 10+, Clang 10+, MSVC 2019+)
- Boost-Bibliotheken (speziell Boost.Math)

#### Abhängigkeiten

```bash
# Ubuntu/Debian
sudo apt-get install libboost-math-dev

# macOS mit Homebrew
brew install boost

# Windows (vcpkg)
vcpkg install boost-math
```

#### Klonen und Kompilieren

```bash
git clone https://github.com/Haasrobertgmxnet/LinearRegression.git
cd LinearRegression

# Mit g++
g++ -std=c++20 -I/pfad/zu/boost -o regression_example LinearRegression.cpp
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
template <class T>
FitResult<T> fit(std::span<const T> x, std::span<const T> y)
```

Passt ein lineares Regressionsmodell an die Daten an.

**Anforderungen:**
- x und y müssen gleiche Größe haben
- Mindestens 3 Datenpunkte erforderlich
- x-Werte dürfen nicht alle identisch sein

#### Funktion: `ci_slope`

```cpp
std::pair<double, double> ci_slope(const FitResult<double>& fitResult, double alpha)
```

Berechnet das Konfidenzintervall für den Steigungskoeffizienten.

**Parameter:**
- `fitResult` - Ergebnis der `fit()`-Funktion
- `alpha` - Signifikanzniveau (z.B. 0.05 für 95% KI)

#### Funktion: `coeff_of_determination`

```cpp
template <typename T>
T coeff_of_determination(const FitResult<T>& fitResult)
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
- **Basado en Templates**: Implementación genérica que soporta varios tipos numéricos (float, double, long double)
- **Integración Boost**: Usa Boost.Math para cuantiles precisos de la distribución t de Student
- **Estadísticas Completas**: Calcula coeficientes de regresión, correlación, SSE e intervalos de confianza
- **Eficiencia de Memoria**: Trabaja con `std::span` para evitar copias innecesarias
- **Seguridad de Tipos**: Uso extensivo de concepts y `static_assert` para seguridad en tiempo de compilación
- **Compatible Header-Only**: Adecuado para integración de biblioteca header-only

### Estructura del Proyecto

```
LinearRegression/
├── LinearRegression/
│   ├── linreg.h      # Implementación principal de regresión
│   ├── stats.h                 # Funciones de utilidad estadística
│   ├── span_compatible.h       # Utilidades helper y concepts
│   └── LinearRegression.cpp               # Ejemplo de uso (si está presente)
├── Documentation/              # Documentación adicional
├── LinearRegression.slnx      # Solución de Visual Studio
└── README.md                  # Este archivo
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
- Compilador compatible con C++20 (GCC 10+, Clang 10+, MSVC 2019+)
- Bibliotecas Boost (específicamente Boost.Math)

#### Dependencias

```bash
# Ubuntu/Debian
sudo apt-get install libboost-math-dev

# macOS con Homebrew
brew install boost

# Windows (vcpkg)
vcpkg install boost-math
```

#### Clonar y Compilar

```bash
git clone https://github.com/Haasrobertgmxnet/LinearRegression.git
cd LinearRegression

# Usando g++
g++ -std=c++20 -I/ruta/a/boost -o regression_example LinearRegression.cpp
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
template <class T>
FitResult<T> fit(std::span<const T> x, std::span<const T> y)
```

Ajusta un modelo de regresión lineal a los datos.

**Requisitos:**
- x e y deben tener el mismo tamaño
- Se requieren mínimo 3 puntos de datos
- Los valores de x no pueden ser todos idénticos

#### Función: `ci_slope`

```cpp
std::pair<double, double> ci_slope(const FitResult<double>& fitResult, double alpha)
```

Calcula el intervalo de confianza para el coeficiente de pendiente.

**Parámetros:**
- `fitResult` - Resultado de la función `fit()`
- `alpha` - Nivel de significancia (ej., 0.05 para IC 95%)

#### Función: `coeff_of_determination`

```cpp
template <typename T>
T coeff_of_determination(const FitResult<T>& fitResult)
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
- **Template-Based**: Generic implementation supporting various numeric types (float, double, long double)
- **Boost Integration**: Uses Boost.Math for accurate Student's t-distribution quantiles
- **Comprehensive Statistics**: Calculates regression coefficients, correlation, SSE, and confidence intervals
- **Memory Efficient**: Works with `std::span` to avoid unnecessary copies
- **Type Safe**: Extensive use of concepts and `static_assert` for compile-time safety
- **Header-Only Compatible**: Suitable for header-only library integration

### Project Structure

```
LinearRegression/
├── LinearRegression/
│   ├── linreg.h      # Main regression implementation
│   ├── stats.h                 # Statistical utility functions
│   ├── span_compatible.h       # Helper utilities and concepts
│   └── LinearRegression.cpp               # Example usage (if present)
├── Documentation/              # Additional documentation
├── LinearRegression.slnx      # Visual Studio solution
└── README.md                  # This file
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
- C++20 compatible compiler (GCC 10+, Clang 10+, MSVC 2019+)
- Boost libraries (specifically Boost.Math)

#### Dependencies

```bash
# Ubuntu/Debian
sudo apt-get install libboost-math-dev

# macOS with Homebrew
brew install boost

# Windows (vcpkg)
vcpkg install boost-math
```

#### Clone and Build

```bash
git clone https://github.com/Haasrobertgmxnet/LinearRegression.git
cd LinearRegression

# Using g++
g++ -std=c++20 -I/path/to/boost -o regression_example LinearRegression.cpp
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
template <class T>
FitResult<T> fit(std::span<const T> x, std::span<const T> y)
```

Fits a linear regression model to the data.

**Requirements:**
- x and y must have the same size
- Minimum 3 data points required
- x values must not all be identical

#### Function: `ci_slope`

```cpp
std::pair<double, double> ci_slope(const FitResult<double>& fitResult, double alpha)
```

Calculates confidence interval for the slope coefficient.

**Parameters:**
- `fitResult` - Result from `fit()` function
- `alpha` - Significance level (e.g., 0.05 for 95% CI)

#### Function: `coeff_of_determination`

```cpp
template <typename T>
T coeff_of_determination(const FitResult<T>& fitResult)
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

### License

This project is open source and available under the MIT License.

### Author

**Haasrobertgmxnet**
- GitHub: [@Haasrobertgmxnet](https://github.com/Haasrobertgmxnet)

### Contributions

Issues and Pull Requests are welcome! Please open an issue for major changes.

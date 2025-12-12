# Model Performance Analysis

### **Training Summary**
- **Epochs:** 200  
- **Final Train Loss:** 1.2783  
- **Final Train Accuracy:** 0.6514  
- **Validation Loss:** 1.3424  
- **Validation Accuracy:** 0.6750  
- **Test Loss:** 1.4251  
- **Test Accuracy:** 0.5992  

---

## Dataset Split Statistics

### **Train Set**
| Species | Count |
|--------|-------|
| Dolphin | 306 |
| Narwhal | 50 |
| Seal | 195 |
| Walrus | 40 |
| Whale | 803 |

### **Validation Set**
| Species | Count |
|--------|-------|
| Dolphin | 228 |
| Whale | 92 |

### **Test Set**
| Species | Count |
|--------|-------|
| Dolphin | 296 |
| Seal | 9 |
| Whale | 194 |

---

## Observations

The results are difficult to interpret due to **significant class imbalance** across the dataset. Some species (e.g., Whale) are heavily over-represented, while others (e.g., Seal, Walrus, Narwhal) are severely under-represented. This likely contributed to unstable training behavior and biased predictions toward majority classes.

Before applying the class-imbalance–aware loss function, the model reached **only ~0.3 accuracy**, which confirms that the imbalance had a major negative impact on performance.

A more robust strategy for addressing imbalance—such as oversampling, undersampling, augmented minority-class generation, or a more advanced loss strategy—would likely improve generalization and class discrimination.

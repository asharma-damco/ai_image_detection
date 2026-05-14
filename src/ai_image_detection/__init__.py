"""
ai_image_detection — Unified AI Image Detection Framework
==========================================================

Combines detection logic from two source pipelines:
  - UAIC fraud detection (vehicle damage + general AI image detection)
  - PIMA FCU ID card edit detection

Public surface
--------------
    from ai_image_detection.detectors.dual_branch import DualBranchDetector
    from ai_image_detection.detectors.trufor      import TruForAnalyzer
    from ai_image_detection.detectors.srm         import SRMAnalyzer
    from ai_image_detection.detectors.siglip2     import SigLIP2Detector
    from ai_image_detection.detectors.clip_ufd    import UFDAdapter
    from ai_image_detection.detectors.dino_probe  import dino_feature_score
    from ai_image_detection.detectors.yolo_damage import DamageDetector
    from ai_image_detection.signals.ela           import ela_anomaly_score
    from ai_image_detection.signals.prnu          import prnu_anomaly_score
    from ai_image_detection.signals.dct_benford   import dct_benford_score
    from ai_image_detection.signals.cfa           import cfa_correlation_score
    from ai_image_detection.signals.metadata      import analyze_metadata
    from ai_image_detection.ensemble.scorer       import EnsembleScorer
    from ai_image_detection.document.classifier   import classify_document
"""

__version__ = "1.0.0"

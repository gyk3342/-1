      マルチユーザー手勢描画システム
          
  本研究では、MediapipeとOpenCVを使用し、マルチユーザーの手勢描画システムを開発した。システムは、複数のユーザーがカメラを用いて手の動きをリアルタイムで描画で
きるようにし、インタラクティブな環境を提供することを目的としている。

システムの設計：
  ビデオキャプチャモジュール：OpenCVを使用してカメラからのビデオストリームを取得。
  手の検出モジュール：Mediapipeを用いて手のランドマークを検出し、手の動きをリアルタイムで追跡。
  描画モジュール：検出された手の座標を利用して、軌跡を描画し、インタラクティブな描画体験を提供。各手に固有の色を割り当て。指の座標に基づいてキャンバス上に線を描画。


応用可能な分野：
  教育：児童向けのインタラクティブな手勢描画活動に応用可能。
  ジェスチャーゲーム：ジェスチャーベースのマルチプレイヤーゲームインタラクションを実装します。


今後の改善点：
  手勢認識の拡張、手を開く、握るといったジェスチャーを認識することで、システムのインタラクションの幅を広げる。
  音声によるヒントやフィードバックを加え、ユーザー体験を強化。
  距離推定の精度を向上させるために、深度カメラの導入を検討。





11.29
  今週の改善点

1.平滑化アルゴリズムの追加：
  deque を使用して指の軌跡位置を平均化する平滑化処理 (smoothing_queue) を導入。
  
2.新しいモード切替機能：
  pen と eraser の2つの描画モードをサポート。
  指ジェスチャー（例：人差し指と中指の距離が短い場合）に基づいて自動的にモードを切り替え。
  また、キーボードの c（キャンバスをクリア）および r（キャンバスを保存）で手動制御が可能。
  
3.描画有効化条件の追加：
  親指と人差し指の距離 (distance_thumb_index) を検出し、描画の有効化/無効化を切り替え。
  距離が特定の閾値未満の場合、描画を無効化し、誤操作を防止。

4.キャンバス管理の改良
  キャンバス背景色：キャンバスの背景色を黒から白に変更。
  キャンバスの拡大率：canvas_size_multiplier を2から3に引き上げ、描画の精度が向上。

5.手の独立識別が最適化
  手を独立して認識し、異なる色を割り当てるよう最適化されています。複数の手での操作がより安定して処理可能に。











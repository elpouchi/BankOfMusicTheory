# Beyblade X Physics Coach

A small Streamlit web app that simulates Beyblade X battles using a
physically-informed 2D rigid body model.  It can coach defensive play,
run Monte-Carlo analyses and optionally analyse uploaded battle videos.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the provided local URL in a browser to use the app.

## Features

* Pick blades, ratchets and bits for two combos.
* Run a single simulated battle and view trajectories.
* Monte-Carlo mode to estimate outcome probabilities with downloadable JSON
  results.
* Live editing of part dictionaries in JSON form.
* Preset buttons for two sample defensive decks.
* (Beta) simple video analysis to estimate spin-down time.
* Utility helpers for exporting deck sheets (PDF/Word).

## Editing parts

Use the **Advanced** expander to modify the `BLADE`, `RATCHET` and `BIT`
dictionaries.  Enter JSON and press *Apply* to hot reload.  You can also
download the current dictionaries or upload a JSON file containing the
three dictionaries in a top-level object.

## Monte-Carlo

Set a seed and number of simulations in the sidebar.  Press
**Run Monte-Carlo** to execute; a bar chart of outcome counts will be
shown and a download button is provided to save the results.

## Known limitations

* The physics model is a coarse 2D approximation and ignores vertical
  dynamics.
* Part statistics are estimated and should be tuned with real-world
  testing.
* Video analysis is heuristic and intended only for quick feedback.

## License

MIT

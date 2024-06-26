
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_jops_examples/chapter_2/plot_air-wind.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_auto_jops_examples_chapter_2_plot_air-wind.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_jops_examples_chapter_2_plot_air-wind.py:


New York air quality data polynomial fits (air quality data)
============================================================

.. GENERATED FROM PYTHON SOURCE LINES 7-47



.. image-sg:: /auto_jops_examples/chapter_2/images/sphx_glr_plot_air-wind_001.png
   :alt: New York air quality
   :srcset: /auto_jops_examples/chapter_2/images/sphx_glr_plot_air-wind_001.png
   :class: sphx-glr-single-img





.. code-block:: Python


    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures


    # Get the data
    data = pd.read_csv("../data/airquality.csv").dropna()
    wind = data["Wind"].to_numpy()
    ozone = data["Ozone"].to_numpy()


    # Least squares linear
    new_wind = np.arange(2, 21, 0.01)
    lm = LinearRegression().fit(wind.reshape(-1, 1), ozone)
    new_ozone_linear = lm.predict(new_wind.reshape(-1, 1))


    # Least squares quadratic
    new_wind = np.arange(2, 21, 0.01)
    qm = make_pipeline(PolynomialFeatures(2), LinearRegression())
    qm.fit(wind.reshape(-1, 1), ozone)
    new_ozone_quadratic = qm.predict(new_wind.reshape(-1, 1))


    # Build the graph
    plt.figure(figsize=(6, 4), dpi=300)
    plt.scatter(wind, ozone, color="#000000", s=2)
    plt.plot(new_wind, new_ozone_linear, color="b", linestyle="dashed")
    plt.plot(new_wind, new_ozone_quadratic, color="r")

    plt.title("New York air quality")
    plt.xlabel("Wind speed (mph)")
    plt.ylabel("Ozone concentration (ppb)")
    plt.grid(linestyle="-", color="#EEEEEE", zorder=0)
    plt.show()


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 8.254 seconds)


.. _sphx_glr_download_auto_jops_examples_chapter_2_plot_air-wind.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_air-wind.ipynb <plot_air-wind.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_air-wind.py <plot_air-wind.py>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_

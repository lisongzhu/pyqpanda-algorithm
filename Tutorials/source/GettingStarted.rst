Installation
=========================

.. _pyqpanda3: https://qcloud.originqc.com.cn/document/qpanda-3/cn/index.html
.. _`Microsoft Visual C++ Redistributable x64`: https://aka.ms/vs/17/release/vc_redist.x64.exe
.. _`pyqpanda_algorithm`: https://originqc.com.cn/zh/quantum_soft.html?type=pyqpanda&lv2id=43&lv3id=221

pyqpanda_alg is an algorithm expansion module based on ``pyqpanda3`` . 
It contains many practical quantum application algorithms. Installation and use need to rely on ``pyqpanda3`` . For the interface usage of ``pyqpanda3`` , please refer to pyqpanda3_

Configuration
>>>>>>>>>>>>>>>>>>>

pyqpanda_alg uses C++ as the host language, and its environmental requirements for the system are as follows:

Windows
---------------------
.. list-table::

    * - software
      - version
    * - `Microsoft Visual C++ Redistributable x64`_ 
      - 2019
    * - Python
      - >= 3.11 && <= 3.13

Linux
---------------------

.. list-table::

    * - software
      - version
    * - GCC
      - >= 7.5 
    * - Python
      - >= 3.11 && <= 3.13


Install
>>>>>>>>>>>>>>>>>

If you have already installed the python environment and the pip tool, enter the following command in the terminal or console:

    .. code-block:: python

        pip install pyqpanda_alg

.. note:: If you encounter permission problems under linux, you need to add ``sudo``
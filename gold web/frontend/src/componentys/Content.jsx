import React, { useState } from 'react'
import { useDispatch } from 'react-redux';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

import '../Styles/form.css'
import { setResult } from '../redux/resultSlice';
import { message } from 'antd';


const Content = () => {

  const [spx, setSpx] = useState();
  const [uso, setUso] = useState();
  const [slv, setSlv] = useState();
  const [euroUsd, setEuroUsd] = useState();

  const [loading, setLoading] = useState();

  const dispatch = useDispatch();

  const navigate = useNavigate();


  const onSubmitForm = async (e) => {

    e.preventDefault();

    const spxNo = Number(spx);
    const usoNo = Number(uso);
    const slvNo = Number(slv);
    const euroUsdNo = Number(euroUsd);

    try {

      setLoading(true);

      const { data } = await axios.post(`${process.env.REACT_APP_BACKEND_URL}/pred-gold-price`, {
        spxFloat: spxNo,
        usoFloat: usoNo,
        slvFloat: slvNo,
        euroUSDRatioFloat: euroUsdNo
      });

      setLoading(false);

      dispatch(setResult(data?.pred_result_value));

      message.success('the price of gold has been predicted successfully')

      navigate('/result');

    } catch (error) {

      setLoading(false);

      message.error('Something Went Wrong!! Please try again.')

      console.log(error);

    };

  }

  return (
    <div className='main-container'>
      <h1 className='header'>Gold Price Prediction</h1>

      <form className='form-container' onSubmit={onSubmitForm}>
        <label htmlFor="fname">SPX:</label><br />
        <input type="number" id="fname" name="fname" placeholder='enter standard-and-poor-500 price' onChange={(e) => setSpx(e.target.value)} required /><br />
        <label htmlFor="lname">USO:</label><br />
        <input type="number" id="lname" name="lname" placeholder='enter united-states-oil-fund price' onChange={(e) => setUso(e.target.value)} required /><br />
        <label htmlFor="lname">SLV:</label><br />
        <input type="number" id="lname" name="lname" placeholder='enter silver price' onChange={(e) => setSlv(e.target.value)} required /><br />
        <label htmlFor="lname">EUR/USD:</label><br />
        <input type="number" id="lname" name="lname" placeholder='enter the ratio of Euro and US Dollar' onChange={(e) => setEuroUsd(e.target.value)} required /><br />
        <br />
        <input className='btn-2' type="submit" disabled={loading ? true : false} value={loading ? "Predicting" : "Submit"} />
      </form>

    </div>
  )
}

export default Content;
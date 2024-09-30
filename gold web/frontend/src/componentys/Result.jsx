import React from 'react'
import { Link } from 'react-router-dom';
import { useSelector } from 'react-redux';

const Result = () => {

  const { resultData } = useSelector((state) => state.resultData);

  console.log(resultData);
  
  return (
    <div className='main-container'>
      <h1 className='header'>Gold Price Prediction Result</h1>

      <form className='form-container'>

       <p className='predicted-price'>Predicted Gold Price: {resultData ? `$ ${resultData}` : 'No Predicted Gold Price Available'}</p>

        <Link to='/content'>
          <input className='btn-2 mar' type="submit" value="Ok !" />
        </Link>
      </form>
    </div>
  )
}

export default Result


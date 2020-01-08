const crypto = require("crypto");
class DigestGenerator {
  /**
   * create a password Digest to be stored in the DB that is compitable with geoserver digest passwords
   * @param {string} password
   * @param {buffer} inputSalt optional salt to use for the hash generation, if not provided a random salt will be used
   */
  static getDigest(password, inputSalt = null) {

    console.log( ' ............... Start of getDigest ...... ' )
    //generate a random salt of 16 bytes if the salt is not present or has a wrong length
    
    //const salt = (inputSalt instanceof Buffer && inputSalt.byteLength === 16) ? inputSalt : crypto.randomBytes(16);
    const salt = Buffer.from(inputSalt)

    //get the password as a buffer (encode it)
    const passwordBuffer = Buffer.from(password);
    //add the salt to the password
    const concated = Buffer.concat([salt, passwordBuffer]);
    //consider the current hash is the same as the salt and password just to use it as input for the hashing loop
    let currentHash = concated;
    //hash the concated password and the salt 100,000 times for each iteration use the output of the last iteration as input
    for (let i = 0; i < 100000; i++) {
      const hash = crypto.createHash("sha256");
      hash.write(currentHash);
      currentHash = hash.digest();
    }

    console.log( ' ............... currentHash = ' + currentHash )
    // add the salt to the hashed password
    const result = Buffer.concat([salt, currentHash]);
    //return the result after adding the prefix digest to indicate that the format of the password is digest

    console.log( ' ............... End of getDigest ...... ' )
    return "digest1:" + result.toString("base64");
  }
  /**
   * verifies a password against a hashed password using the digest string that is stored in the DB
   * @param {string} digestStr
   * @param {string} password
   */
  static verifyPassword(digestStr, password) {
    //get the salt
    const salt = this.getBuffers(digestStr).salt;
    // make a digest for the input password for the salt from the digest string (the salt of the original password)
    const result = this.getDigest(password, salt);
    //compare the generated digest for the given password to the provided correct disgest
    return result === digestStr;
  }
  /**
   * split the digest string and get the salt and hash sperated as buffers
   * @param {string} digestStr
   */
  static getBuffers(digestStr) {

    console.log( ' ............... Start of getBuffers ...... ' )

    //get base 64 string contains the salt and the hashed password only
    //first 16 bytes are for salt the next 32 bytes are for the hashed password
    const pureDigestStr = digestStr.replace("digest1:", "");
    const buf = Buffer.from(pureDigestStr, "base64");

    console.log( '  .... buf = ' + buf )
    //slice the salt
    const salt = buf.slice(0, 16);
    const hashedPassword = buf.slice(16);

    console.log( ' ............... End of getBuffers ...... ' )
    return {
      salt: salt,
      hash: hashedPassword
    };
  }
  /**
   * claculates a response for a challenge using the digest string in from the DB which contains the salt and the hashed password
   * @param {string} digestStr base 64 representation of the salt and the hashed password in the DB
   * @param {Buffer} challenge buffer that contains the challenge
   */
  static getChallengeResponse(digestStr, challenge) {
    
    console.log( ' ............... Start of getChallengeResponse ...... ' )
    
    const {
      salt, hash
    } = this.getBuffers(digestStr);

    console.log( ' Salt = ' + salt)
    console.log( ' hash = ' + hash)
    let currentHash = Buffer.concat([salt, Buffer.from(challenge), hash]);
    console.log( ' currentHash = ' + currentHash)
    console.log( currentHash.toString('base64') )
    //hash the concated password and the salt 100,000 times for each iteration use the output of the last iteration as input

    console.log(' ***** awais *****')


    for (let i = 0; i < 5; i++) {
      const hashAlg = crypto.createHash("sha256");
      hashAlg.write(currentHash);
      currentHash = hashAlg.digest();
    }

    console.log( ' ......... currentHash = ' + currentHash)
    console.log(  currentHash.toString('base64'))
    console.log( ' ............... End of getChallengeResponse ......' )
    
    return currentHash;
  }
  /**
   * calculates a challenge response using a salt and challenge and plain text password
   * @param {Buffer} salt
   * @param {Buffer} challenge
   * @param {string} password
   */
  static calcChallengeResponse(salt, challenge, password) {
    const digestStr = this.getDigest(password, salt);
    console.log(digestStr)
    return this.getChallengeResponse(digestStr, challenge);
  }
  /**
   *
   * @param {string} digestStr base 64 representation of the salt and the hashed password in the DB
   * @param {Buffer} challenge buffer that contains the challenge
   * @param {string} clientResponse base 64 representation of the calculated response from the client to the challenge sent from the server with the salt
   */
  static verifyChallengeResponse(digestStr, challenge, clientResponse) {
    const correctResponse = this.getChallengeResponse(digestStr, challenge);
    console.log(correctResponse.toString("base64"));
    console.log(clientResponse.toString("base64"));
    return correctResponse.toString("base64") === clientResponse;
  }
}
var result = DigestGenerator.calcChallengeResponse("j3zpl6wHbivcG2phZsw8Kw==","s7lcwRFS/WiZA94LgXMxo8mPDX2EdOcoWFZwleaTbIE=","lK98hgr&h")































//https://repl.it/languages/nodejs

